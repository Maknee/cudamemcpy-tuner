"""Unit tests for kernel_patches/nvidia-open/kmod.sh's pure logic.

No root, GPU, NVIDIA module or network is used: the Python program embedded in
kmod.sh is extracted to a temporary module and imported (its main block does not
run), and every function that would read or change loaded modules is replaced.
Exit code 77 means skipped (not Linux).
"""
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import stat
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / 'kernel_patches' / 'nvidia-open'
SCRIPT = KERNEL_DIR / 'kmod.sh'
OPENER = 'exec python3 -I - "$@" <<\'PY\'\n'


def load_kmod(directory):
    text = SCRIPT.read_text()
    start = text.index(OPENER) + len(OPENER)
    end = text.index('\nPY\n', start)
    source = Path(directory) / 'kmod_under_test.py'
    source.write_text(text[start:end + 1])
    with mock.patch.dict(os.environ, {'CMT_KERNEL_SCRIPT_DIR': str(KERNEL_DIR)}):
        spec = importlib.util.spec_from_file_location('kmod_under_test', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


ORIGINAL = {'nvidia': {'version': '615.71.09', 'srcversion': 'A', 'build_id_note': None,
                       'parameters': {'NVreg_RegistryDwords': '(null)'}},
            'nvidia_uvm': None}
PATCHED = {'nvidia': {'version': '615.71.09', 'srcversion': 'B', 'build_id_note': None,
                      'parameters': {'NVreg_RegistryDwords': 'RMAllowSysmemLargePages=1'}},
           'nvidia_uvm': None}


class KmodTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module_directory = tempfile.TemporaryDirectory(prefix='kmod-module-')
        cls.addClassCleanup(cls.module_directory.cleanup)
        cls.kmod = load_kmod(cls.module_directory.name)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='kmod-test-')
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def patch(self, name, value):
        patcher = mock.patch.object(self.kmod, name, value)
        patcher.start()
        self.addCleanup(patcher.stop)

    def write_journal(self, **fields):
        state = {'schema': 'cmt.kernel-state', 'version': 1, 'kernel': self.kmod.KERNEL,
                 'boot_id': self.kmod.BOOT_ID, 'release': '615.71.09', 'original': ORIGINAL,
                 'observed': PATCHED, 'restore_set': str(self.directory / 'original-1'),
                 'measurement_lock': str(self.directory / 'lock'), 'original_arguments': {}}
        state.update(fields)
        path = self.directory / 'state.json'
        path.write_text(json.dumps(state))
        return path, state

    def archived(self, kind):
        return sorted(self.directory.glob(f'state-{kind}-*.json'))


class ExistingJournalTests(KmodTestCase):
    """patch's decision about a journal left by an earlier patch."""

    def test_restored_journal_is_archived_in_any_boot(self):
        for boot in (self.kmod.BOOT_ID, 'another-boot'):
            path, _ = self.write_journal(status='restored', boot_id=boot)
            self.patch('loaded', lambda: self.fail('a restored journal needs no module check'))
            self.assertIsNone(self.kmod.existing_journal(path))
            self.assertFalse(path.exists())
        self.assertEqual(len(self.archived('restored')), 2)

    def test_active_in_this_boot(self):
        path, _ = self.write_journal(status='active')
        self.patch('loaded', lambda: PATCHED)
        self.assertEqual(self.kmod.existing_journal(path), 'active')
        self.assertTrue(path.exists())

    def test_active_in_this_boot_but_changed(self):
        path, _ = self.write_journal(status='active')
        self.patch('loaded', lambda: ORIGINAL)
        with self.assertRaisesRegex(self.kmod.Refusal, 'loaded state changed'):
            self.kmod.existing_journal(path)
        self.assertTrue(path.exists())

    def test_unfinished_in_this_boot_is_refused(self):
        for status in ('prepared', 'switching', 'patch_failed', 'restoring', 'restore_failed'):
            path, _ = self.write_journal(status=status)
            self.patch('loaded', lambda: ORIGINAL)
            with self.assertRaisesRegex(self.kmod.Refusal, 'unfinished recovery journal'):
                self.kmod.existing_journal(path)
            self.assertTrue(path.exists())

    def test_previous_boot_with_original_modules_is_archived_as_stale(self):
        for status in ('active', 'switching', 'restore_failed'):
            path, _ = self.write_journal(status=status, boot_id='previous-boot')
            self.patch('loaded', lambda: ORIGINAL)
            with mock.patch('sys.stderr', new=open(os.devnull, 'w')) as sink:
                self.addCleanup(sink.close)
                self.assertIsNone(self.kmod.existing_journal(path))
            self.assertFalse(path.exists())
        stale = self.archived('stale')
        self.assertEqual(len(stale), 3)
        self.assertEqual(json.loads(stale[0].read_text())['boot_id'], 'previous-boot')

    def test_previous_boot_with_other_modules_is_refused_with_instructions(self):
        path, _ = self.write_journal(status='active', boot_id='previous-boot')
        self.patch('loaded', lambda: PATCHED)
        with self.assertRaises(self.kmod.Refusal) as caught:
            self.kmod.existing_journal(path)
        message = str(caught.exception)
        self.assertIn('another boot', message)
        self.assertIn(f'mv {path} ', message)
        self.assertIn('cmt kernel inspect', message)
        self.assertTrue(path.exists())
        self.assertEqual(self.archived('stale'), [])

    def test_not_a_journal(self):
        path = self.directory / 'state.json'
        path.write_text('[]')
        with self.assertRaisesRegex(self.kmod.Refusal, 'not a cmt kernel journal'):
            self.kmod.existing_journal(path)


class UnpatchTests(KmodTestCase):
    """unpatch uses the journal's own records and handles a journal from another boot."""

    def setUp(self):
        super().setUp()
        self.rollbacks = []
        self.patch('protected_path', lambda path: Path(path))
        self.patch('measurement_owner', lambda path: contextlib.nullcontext())
        self.patch('mutation_owner', lambda: contextlib.nullcontext())
        self.patch('rollback', lambda state, path: self.rollbacks.append(state['status']))

    def unpatch(self, path):
        args = types.SimpleNamespace(state=str(path), measurement_lock=None)
        with mock.patch('sys.stdout', new=open(os.devnull, 'w')) as out, \
                mock.patch('sys.stderr', new=open(os.devnull, 'w')) as err:
            self.addCleanup(out.close)
            self.addCleanup(err.close)
            return self.kmod.unpatch(args)

    def test_active_journal_rolls_back(self):
        path, _ = self.write_journal(status='active')
        self.patch('loaded', lambda: PATCHED)
        self.assertEqual(self.unpatch(path), 0)
        self.assertEqual(self.rollbacks, ['active'])

    def test_previous_boot_with_original_modules(self):
        path, _ = self.write_journal(status='active', boot_id='previous-boot')
        self.patch('loaded', lambda: ORIGINAL)
        self.assertEqual(self.unpatch(path), 0)
        self.assertEqual(self.rollbacks, [])
        self.assertFalse(path.exists())
        self.assertEqual(len(self.archived('stale')), 1)

    def test_previous_boot_with_other_modules(self):
        path, _ = self.write_journal(status='switching', boot_id='previous-boot')
        self.patch('loaded', lambda: PATCHED)
        with self.assertRaisesRegex(self.kmod.Refusal, 'another boot'):
            self.unpatch(path)
        self.assertEqual(self.rollbacks, [])
        self.assertTrue(path.exists())

    def test_explicit_measurement_lock_overrides_the_journal(self):
        path, _ = self.write_journal(status='restored')
        self.patch('loaded', lambda: ORIGINAL)
        seen = []
        self.patch('measurement_owner', lambda lock: seen.append(lock) or contextlib.nullcontext())
        args = types.SimpleNamespace(state=str(path), measurement_lock='/elsewhere/lock')
        with mock.patch('sys.stdout', new=open(os.devnull, 'w')) as out:
            self.addCleanup(out.close)
            self.assertEqual(self.kmod.unpatch(args), 0)
        self.assertEqual(seen, ['/elsewhere/lock'])

    def test_journal_release_does_not_need_the_catalog(self):
        self.patch('SCRIPT_DIR', self.directory)  # no releases.json here
        self.kmod.select_journal_release({'release': '999.1.2'})
        self.assertEqual((self.kmod.RELEASE, self.kmod.RELEASE_ENTRY), ('999.1.2', {}))
        for bad in ({}, {'release': '../x'}, {'release': 1}, []):
            with self.assertRaises(self.kmod.Refusal):
                self.kmod.select_journal_release(bad)


class RollbackDecisionTests(KmodTestCase):
    def setUp(self):
        super().setUp()
        self.path, self.state = self.write_journal(status='prepared')
        self.rollbacks = []
        self.patch('rollback', lambda state, path: self.rollbacks.append(dict(state)))

    def test_failure_before_switching_does_not_restore(self):
        def fail(state, path, large):
            raise self.kmod.Refusal('idle check failed')
        self.patch('switch_set', fail)
        with self.assertRaises(self.kmod.Refusal):
            self.kmod.switch_or_rollback(self.state, self.path)
        self.assertEqual(self.rollbacks, [])
        self.assertEqual(json.loads(self.path.read_text())['status'], 'patch_failed')

    def test_failure_after_switching_restores(self):
        for error in (self.kmod.Refusal('insmod failed'), InterruptedError('interrupted by signal 1'),
                      KeyboardInterrupt()):
            self.rollbacks.clear()

            def fail(state, path, large):
                self.kmod.journal(state, path, 'switching', selected_arguments={})
                raise error
            self.patch('switch_set', fail)
            with self.assertRaises(type(error)):
                self.kmod.switch_or_rollback(self.state, self.path)
            self.assertEqual(len(self.rollbacks), 1)
            self.assertEqual(self.rollbacks[0]['status'], 'patch_failed')


class SignalTests(KmodTestCase):
    def test_rollback_ignores_interrupts_and_hangup_while_restoring(self):
        path, state = self.write_journal(status='switching', selected_arguments={})
        saved = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
        self.addCleanup(lambda: [signal.signal(number, handler) for number, handler in saved.items()])
        seen = {}

        def restore(state, path):
            seen.update({number: signal.getsignal(number) for number in saved})
        self.patch('restore_state', restore)
        self.kmod.rollback(state, path)
        self.assertEqual(seen, {number: signal.SIG_IGN for number in saved})

    def test_hangup_is_handled_unless_already_ignored(self):
        saved = signal.getsignal(signal.SIGHUP)
        self.addCleanup(signal.signal, signal.SIGHUP, saved)
        signal.signal(signal.SIGHUP, signal.SIG_DFL)
        self.assertIn(signal.SIGHUP, self.kmod.forwarded_signals())
        signal.signal(signal.SIGHUP, signal.SIG_IGN)  # e.g. nohup
        self.assertNotIn(signal.SIGHUP, self.kmod.forwarded_signals())
        self.assertIn(signal.SIGTERM, self.kmod.forwarded_signals())

    def test_signal_handler_raises(self):
        with self.assertRaises(InterruptedError):
            self.kmod.interrupted(signal.SIGHUP, None)


class MeasurementLockTests(KmodTestCase):
    def scratch_places(self):
        """Point /run/user and /tmp at scratch directories (production never changes them; tests cannot)."""
        runtime = self.directory / 'run-user'
        fallback = self.directory / 'tmp'
        runtime.mkdir()
        fallback.mkdir()
        self.patch('RUNTIME_BASE', str(runtime))
        self.patch('FALLBACK_BASE', str(fallback))
        return runtime, fallback

    def test_production_places_are_fixed(self):
        self.assertEqual(self.kmod.RUNTIME_BASE, '/run/user')
        self.assertEqual(self.kmod.FALLBACK_BASE, '/tmp')

    def test_default_path(self):
        runtime, fallback = self.scratch_places()
        for uid in (os.geteuid(), 1234, 1000):
            (runtime / str(uid)).mkdir(exist_ok=True)
        environment = {key: value for key, value in os.environ.items()
                       if key not in ('CMT_MEASUREMENT_LOCK_PATH', 'SUDO_UID')}
        with mock.patch.dict(os.environ, environment, clear=True):
            self.assertEqual(self.kmod.default_measurement_lock(),
                             f'{runtime}/{os.geteuid()}/cmt-measurements.lock')
            with mock.patch.dict(os.environ, {'SUDO_UID': '1234'}):
                with mock.patch.object(self.kmod.os, 'geteuid', return_value=0):
                    self.assertEqual(self.kmod.default_measurement_lock(), f'{runtime}/1234/cmt-measurements.lock')
                with mock.patch.object(self.kmod.os, 'geteuid', return_value=1000):
                    self.assertEqual(self.kmod.default_measurement_lock(), f'{runtime}/1000/cmt-measurements.lock')
            with mock.patch.dict(os.environ, {'CMT_MEASUREMENT_LOCK_PATH': 'relative.lock'}):
                self.assertEqual(self.kmod.default_measurement_lock(), os.path.abspath('relative.lock'))
            # Without /run/user/<uid>: the private directory under /tmp.
            (runtime / str(os.geteuid())).rmdir()
            self.assertEqual(self.kmod.default_measurement_lock(),
                             f'{fallback}/cmt-{os.geteuid()}/cmt-measurements.lock')

    def test_runtime_directory_is_used_when_present(self):
        runtime, fallback = self.scratch_places()
        (runtime / str(os.geteuid())).mkdir(mode=0o700)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('CMT_MEASUREMENT_LOCK_PATH', None)
            path = self.kmod.default_measurement_lock()
            with self.kmod.measurement_owner(path):
                self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
        self.assertEqual(list(fallback.iterdir()), [])

    def test_fallback_directory_is_created_private(self):
        runtime, fallback = self.scratch_places()
        directory = fallback / f'cmt-{os.geteuid()}'
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('CMT_MEASUREMENT_LOCK_PATH', None)
            path = self.kmod.default_measurement_lock()
            self.assertEqual(path, str(directory / 'cmt-measurements.lock'))
            mask = os.umask(0o077)  # mkdir's mode is masked; the directory must still end up 0700
            try:
                with self.kmod.measurement_owner(path):
                    with self.assertRaisesRegex(self.kmod.Refusal, 'another measurement owns'):
                        self.kmod.measurement_owner(path)
            finally:
                os.umask(mask)
            self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE((directory / 'cmt-measurements.lock').stat().st_mode), 0o600)
            with self.kmod.measurement_owner(path):  # the existing private directory is reused
                pass

    def test_fallback_directory_of_another_user_is_refused(self):
        _, fallback = self.scratch_places()
        uid = os.geteuid() + 1  # the test cannot make another user's directory; it asks for another user's lock
        directory = fallback / f'cmt-{uid}'
        directory.mkdir(mode=0o700)
        self.patch('measurement_uid', lambda: uid)
        with self.assertRaisesRegex(self.kmod.Refusal, 'owned by uid'):
            self.kmod.measurement_owner(str(directory / 'cmt-measurements.lock'))
        self.assertEqual(list(directory.iterdir()), [])

    def test_fallback_directory_others_can_read_is_refused(self):
        _, fallback = self.scratch_places()
        directory = fallback / f'cmt-{os.geteuid()}'
        directory.mkdir()
        directory.chmod(0o755)
        with self.assertRaisesRegex(self.kmod.Refusal, 'mode 0700'):
            self.kmod.measurement_owner(str(directory / 'cmt-measurements.lock'))
        self.assertEqual(list(directory.iterdir()), [])
        self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o755)  # reported, never repaired

    def test_symlinked_fallback_directory_is_refused(self):
        _, fallback = self.scratch_places()
        real = self.directory / 'real'
        real.mkdir(mode=0o700)
        (fallback / f'cmt-{os.geteuid()}').symlink_to(real)
        with self.assertRaisesRegex(self.kmod.Refusal, 'symlink'):
            self.kmod.measurement_owner(str(fallback / f'cmt-{os.geteuid()}' / 'cmt-measurements.lock'))
        self.assertEqual(list(real.iterdir()), [])

    def test_symlinked_lock_in_fallback_directory_is_refused(self):
        _, fallback = self.scratch_places()
        directory = fallback / f'cmt-{os.geteuid()}'
        directory.mkdir(mode=0o700)
        directory.chmod(0o700)
        target = self.directory / 'target'
        (directory / 'cmt-measurements.lock').symlink_to(target)
        with self.assertRaisesRegex(self.kmod.Refusal, 'without following symlinks'):
            self.kmod.measurement_owner(str(directory / 'cmt-measurements.lock'))
        self.assertFalse(target.exists())

    def test_creates_and_holds(self):
        path = self.directory / 'cmt-measurements.lock'
        with self.kmod.measurement_owner(str(path)):
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            with self.assertRaisesRegex(self.kmod.Refusal, 'another measurement owns'):
                self.kmod.measurement_owner(str(path))
        with self.kmod.measurement_owner(str(path)):
            pass

    def test_symlinked_lock_is_refused(self):
        target = self.directory / 'target'
        target.write_text('')
        (self.directory / 'link.lock').symlink_to(target)
        with self.assertRaisesRegex(self.kmod.Refusal, 'without following symlinks'):
            self.kmod.measurement_owner(str(self.directory / 'link.lock'))
        dangling = self.directory / 'dangling.lock'
        dangling.symlink_to(self.directory / 'absent')
        with self.assertRaises(self.kmod.Refusal):
            self.kmod.measurement_owner(str(dangling))
        self.assertFalse((self.directory / 'absent').exists())

    def test_symlinked_directory_is_refused(self):
        real = self.directory / 'real'
        real.mkdir()
        (self.directory / 'runtime').symlink_to(real)
        with self.assertRaisesRegex(self.kmod.Refusal, 'symlink'):
            self.kmod.measurement_owner(str(self.directory / 'runtime' / 'cmt-measurements.lock'))
        self.assertEqual(list(real.iterdir()), [])

    def test_missing_directory_is_refused_not_created(self):
        missing = self.directory / 'run-user-4242'
        with self.assertRaisesRegex(self.kmod.Refusal, 'does not exist'):
            self.kmod.measurement_owner(str(missing / 'cmt-measurements.lock'))
        self.assertFalse(missing.exists())

    def test_foreign_owner_is_refused(self):
        if os.geteuid() == 0:
            self.skipTest('the file would be owned by root, which is accepted')
        path = self.directory / 'cmt-measurements.lock'
        path.write_text('')
        self.patch('measurement_uid', lambda: os.geteuid() + 1)
        with self.assertRaisesRegex(self.kmod.Refusal, 'is owned by uid'):
            self.kmod.measurement_owner(str(path))

    def test_non_regular_file_is_refused_without_blocking(self):
        fifo = self.directory / 'fifo.lock'
        os.mkfifo(fifo)
        with self.assertRaisesRegex(self.kmod.Refusal, 'regular lock file'):
            self.kmod.measurement_owner(str(fifo))


class MutationLockTests(KmodTestCase):
    def test_requires_root(self):
        if os.geteuid() == 0:
            self.skipTest('running as root')
        with self.assertRaisesRegex(self.kmod.Refusal, 'requires root'):
            self.kmod.mutation_owner()

    def test_lock_directory_must_be_root_owned_0700(self):
        if os.geteuid() == 0:
            self.skipTest('running as root')
        # Created by a non-root user: exists afterwards but is not root-owned.
        with self.assertRaisesRegex(self.kmod.Refusal, 'root-owned 0700 directory'):
            self.kmod.root_directory(str(self.directory), 'cmt')
        self.assertEqual(stat.S_IMODE((self.directory / 'cmt').stat().st_mode), 0o700)
        (self.directory / 'file').write_text('')
        with self.assertRaisesRegex(self.kmod.Refusal, 'root-owned 0700 directory'):
            self.kmod.root_directory(str(self.directory), 'file')
        (self.directory / 'real').mkdir(mode=0o700)
        (self.directory / 'link').symlink_to(self.directory / 'real')
        with self.assertRaisesRegex(self.kmod.Refusal, 'root-owned 0700 directory'):
            self.kmod.root_directory(str(self.directory), 'link')

    def test_lock_location(self):
        self.assertEqual(os.path.join(self.kmod.MUTATION_LOCK_PARENT, self.kmod.MUTATION_LOCK_DIRECTORY,
                                      self.kmod.MUTATION_LOCK_NAME), '/run/cmt/kernel.lock')


class PathTests(KmodTestCase):
    def test_root_work_dir_is_not_in_a_home_directory(self):
        with mock.patch.object(self.kmod.os, 'geteuid', return_value=0):
            self.assertEqual(self.kmod.default_work_dir(), '/var/cache/cmt/kernel')
        with mock.patch.object(self.kmod.os, 'geteuid', return_value=1000):
            self.assertEqual(self.kmod.default_work_dir(), str(Path.home() / '.cache/cmt/kernel'))

    def test_protected_directory_checks_before_creating(self):
        if os.geteuid() == 0:
            self.skipTest('running as root')
        target = self.directory / 'a' / 'b'
        with self.assertRaisesRegex(self.kmod.Refusal, 'root-owned'):
            self.kmod.protected_directory(target)
        self.assertFalse((self.directory / 'a').exists())

    def test_arguments_become_absolute_before_leaving_the_directory(self):
        previous = os.getcwd()
        self.addCleanup(os.chdir, previous)
        os.chdir(self.directory)
        args = types.SimpleNamespace(action='patch', work_dir='work', state='state.json', restore_set=None,
                                     source='src', measurement_lock='lock')
        self.kmod.prepare_arguments(args)
        self.assertEqual(os.getcwd(), '/')
        self.assertEqual(self.kmod.ORIGINAL_CWD, str(self.directory))
        for name, value in (('work_dir', 'work'), ('state', 'state.json'), ('source', 'src'),
                            ('measurement_lock', 'lock')):
            self.assertEqual(getattr(args, name), str(self.directory / value))
        self.assertIsNone(args.restore_set)


class CatalogTests(KmodTestCase):
    def test_minimum_kernel(self):
        self.assertEqual(self.kmod.kernel_version('6.6.36'), (6, 6))
        self.assertEqual(self.kmod.kernel_version('5.15.0-105-generic'), (5, 15))
        self.patch('MINIMUM_KERNEL', '5.8')
        for kernel, accepted in (('5.4.0-150-generic', False), ('5.7.19', False), ('5.8.0', True),
                                 ('5.10.0', True), ('6.6.36', True)):
            self.patch('KERNEL', kernel)
            if accepted:
                self.kmod.require_minimum_kernel()
            else:
                with self.assertRaisesRegex(self.kmod.Refusal, 'older than 5.8'):
                    self.kmod.require_minimum_kernel()

    def test_every_release_matches_its_patch_files(self):
        catalog = json.loads((KERNEL_DIR / 'releases.json').read_text())
        self.assertEqual(catalog['minimum_kernel'], '5.8')
        self.assertEqual(len(catalog['releases']), 59)
        for release, entry in catalog['releases'].items():
            self.kmod.select_release(release)
            self.assertEqual(self.kmod.MINIMUM_KERNEL, '5.8')
            files = self.kmod.patch_files()
            self.assertEqual(len(files), 3, release)
            for record, path in files:
                text = path.read_text()
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record['sha256'])
                self.assertTrue(text.startswith('SPDX-License-Identifier: GPL-2.0-only\n'), path)
                if path.name.startswith(('a_', 'b_')):
                    self.assertIn('#error "host-registration patch', text, path)
            self.assertNotIn('patch_application_log_sha256', entry['verification'])


if __name__ == '__main__':
    if not sys.platform.startswith('linux'):
        sys.exit(77)
    unittest.main()
