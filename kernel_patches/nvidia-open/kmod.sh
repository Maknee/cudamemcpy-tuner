#!/usr/bin/env bash
# One explicit NVIDIA module owner. Python's standard library supplies checked
# argv execution and atomic JSON journals; no shell command is evaluated.
set -euo pipefail
# This runs as root: use only system tools. `python3 -I` (isolated mode) ignores
# PYTHON* variables and user site-packages and does not import from the current
# directory. Python makes user-supplied relative paths absolute and then
# changes to / before running any tool.
export PATH=/usr/sbin:/usr/bin:/sbin:/bin
CMT_KERNEL_SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
export CMT_KERNEL_SCRIPT_DIR
exec 9<&0
exec python3 -I - "$@" <<'PY'
import argparse
import contextlib
import fcntl
import glob
import gzip
import hashlib
import json
import lzma
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import time

# Restore the caller's stdin after the interpreter has consumed this heredoc.
# (Unit tests import this program as a module; they skip the main block.)
if __name__ == '__main__':
    os.dup2(9, 0)
    os.close(9)

MODULES = {'nvidia': 'nvidia.ko', 'nvidia_uvm': 'nvidia-uvm.ko'}
SCRIPT_DIR = Path(os.environ['CMT_KERNEL_SCRIPT_DIR'])
BUNDLE = 'host-registration'
CAP_SYS_MODULE = 16
SOURCE_URL = 'https://github.com/NVIDIA/open-gpu-kernel-modules.git'
DEFAULT_STATE = '/var/lib/cmt/kernel/state.json'
# Root-owned 0700 directory holding the machine-wide kernel mutation lock.
MUTATION_LOCK_PARENT, MUTATION_LOCK_DIRECTORY, MUTATION_LOCK_NAME = '/run', 'cmt', 'kernel.lock'
KERNEL = os.uname().release
BOOT_ID = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
RELEASE = None
RELEASE_ENTRY = None
MINIMUM_KERNEL = None
# The directory the command was started from; Python changes to / after
# making path arguments absolute. `kernel run` starts its command here.
ORIGINAL_CWD = None

class Refusal(Exception):
    pass

def command(argv, *, cwd=None, check=True):
    result = subprocess.run([str(a) for a in argv], cwd=cwd, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and result.returncode:
        raise Refusal(f'{argv[0]} failed ({result.returncode}): {result.stderr.strip()}')
    return result

def detect_release():
    path = Path('/sys/module/nvidia/version')
    if not path.is_file():
        raise Refusal('nvidia is not loaded; cannot select the running NVIDIA driver release')
    release = path.read_text().strip()
    # NVIDIA also publishes two-part releases (570.144, 580.142).
    if not re.fullmatch(r'\d+\.\d+(?:\.\d+)?', release):
        raise Refusal(f'unrecognized NVIDIA module release: {release!r}')
    return release

def select_release(release):
    global RELEASE, RELEASE_ENTRY, MINIMUM_KERNEL
    catalog = read_json(SCRIPT_DIR / 'releases.json')
    if (catalog.get('schema') != 'cmt.kernel-releases' or catalog.get('version') != 1
            or catalog.get('bundle') != BUNDLE):
        raise Refusal('unsupported kernel release catalog')
    minimum = catalog.get('minimum_kernel')
    if not isinstance(minimum, str) or not re.fullmatch(r'\d+\.\d+', minimum):
        raise Refusal('kernel release catalog lacks a valid minimum_kernel')
    entry = catalog.get('releases', {}).get(release)
    if not isinstance(entry, dict):
        raise Refusal(f'NVIDIA {release} has no kernel patch catalog entry')
    RELEASE, RELEASE_ENTRY, MINIMUM_KERNEL = release, entry, minimum

def select_journal_release(state):
    # Restoration uses only the journal's own records (restore set, manifest,
    # arguments), never the current catalog: a catalog that no longer lists the
    # patched release must not block returning to the original modules.
    global RELEASE, RELEASE_ENTRY
    release = state.get('release') if isinstance(state, dict) else None
    if not isinstance(release, str) or not re.fullmatch(r'\d+\.\d+(?:\.\d+)?', release):
        raise Refusal('recovery journal does not record a valid NVIDIA release')
    RELEASE, RELEASE_ENTRY = release, {}

def kernel_version(release):
    found = re.match(r'(\d+)\.(\d+)', release)
    if not found:
        raise Refusal(f'cannot parse the running kernel release {release!r}')
    return int(found[1]), int(found[2])

def require_minimum_kernel():
    minimum = tuple(int(part) for part in MINIMUM_KERNEL.split('.'))
    if kernel_version(KERNEL) < minimum:
        raise Refusal(f'Linux {KERNEL} is older than {MINIMUM_KERNEL}, the minimum kernel releases.json '
                      f'declares for the {BUNDLE} patches; nothing was built or loaded')

def patch_files():
    if RELEASE_ENTRY.get('status') != 'apply_checked':
        raise Refusal(f'NVIDIA {RELEASE} is not patchable: {RELEASE_ENTRY.get("reason", "source review pending")}')
    commit = RELEASE_ENTRY.get('source_commit', '')
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise Refusal('release catalog lacks an exact official source commit')
    patches = RELEASE_ENTRY.get('patches', [])
    if not patches:
        raise Refusal('release catalog has no checked patch bundle')
    result = []
    for entry in patches:
        path = (SCRIPT_DIR / entry['path']).resolve(strict=True)
        if SCRIPT_DIR.resolve() not in path.parents or sha(path) != entry['sha256']:
            raise Refusal(f'patch path/hash differs from the release catalog: {entry["path"]}')
        if entry.get('directory', '') not in ('', 'kernel-open'):
            raise Refusal('unsupported patch source directory')
        result.append((entry, path))
    return result

def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def read_json(path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise Refusal(f'duplicate JSON key: {key}')
            value[key] = item
        return value
    with open(path) as file:
        return json.load(file, object_pairs_hook=unique)

def atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.kernel-', delete=False) as file:
        temporary = Path(file.name)
        json.dump(value, file, indent=2, sort_keys=True)
        file.write('\n')
        file.flush()
        os.fsync(file.fileno())
        os.fchmod(file.fileno(), 0o644)
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

def module_field(path, field):
    return command(['modinfo', '-F', field, path]).stdout.strip()

def module_build_id(path):
    notes = command(['readelf', '-n', path]).stdout
    build_id = re.search(r'Build ID: ([0-9a-fA-F]+)', notes)
    return build_id[1].lower() if build_id else None

def module_evidence(path):
    fields = {key: module_field(path, key) for key in ('name', 'version', 'srcversion', 'vermagic')}
    fields['build_id'] = module_build_id(path)
    fields['sha256'] = sha(path)
    return fields

def note_build_id(note):
    if note is None:
        return None
    data = bytes.fromhex(note)
    if len(data) < 12:
        raise Refusal('truncated loaded module GNU build-id note')
    name_size, value_size, kind = struct.unpack_from('<III' if sys.byteorder == 'little' else '>III', data)
    offset = 12 + ((name_size + 3) & ~3)
    if kind != 3 or data[12:12 + name_size] != b'GNU\0' or offset + value_size > len(data):
        raise Refusal('unrecognized loaded module GNU build-id note')
    return data[offset:offset + value_size].hex()

def loaded():
    result = {}
    for name in MODULES:
        directory = Path('/sys/module') / name
        if not directory.is_dir():
            result[name] = None
            continue
        parameters = {}
        for path in sorted((directory / 'parameters').glob('*')):
            parameters[path.name] = path.read_text().rstrip('\n')
        result[name] = {
            'version': (directory / 'version').read_text().strip() if (directory / 'version').exists() else None,
            'srcversion': (directory / 'srcversion').read_text().strip(),
            'build_id_note': (directory / 'notes/.note.gnu.build-id').read_bytes().hex()
                if (directory / 'notes/.note.gnu.build-id').exists() else None,
            'parameters': parameters,
        }
    return result

def elf_identity(path):
    with open(path, 'rb') as file:
        header = file.read(20)
    if len(header) != 20 or header[:4] != b'\x7fELF' or header[4] not in (1, 2) or header[5] not in (1, 2):
        raise Refusal(f'not a supported ELF library: {path}')
    order = 'little' if header[5] == 1 else 'big'
    return {'class': 32 if header[4] == 1 else 64,
            'machine': int.from_bytes(header[18:20], order),
            'type': int.from_bytes(header[16:18], order), 'byte_order': order}

def userspace():
    # A custom LD_LIBRARY_PATH would make loader-cache evidence ambiguous.
    output = command(['ldconfig', '-p']).stdout
    libraries = {}
    ignored = []
    native_machine = {'x86_64': 62, 'aarch64': 183}.get(os.uname().machine)
    if native_machine is None:
        raise Refusal('kernel workflow only supports native x86-64 or AArch64 ELF libraries')
    for soname, basename in [('libcuda.so.1', 'libcuda.so.'), ('libnvidia-ml.so.1', 'libnvidia-ml.so.')]:
        candidates = []
        for line in output.splitlines():
            if re.search(r'^\s*' + re.escape(soname) + r'\s', line) and '=>' in line:
                candidates.append(Path(line.split('=>', 1)[1].strip()).resolve())
        compatible = []
        for path in sorted(set(candidates)):
            identity = elf_identity(path)
            if identity['class'] == 64 and identity['machine'] == native_machine and identity['type'] == 3:
                compatible.append(path)
            else:
                ignored.append({'soname': soname, 'path': str(path), 'elf': identity})
        if len(compatible) != 1:
            raise Refusal(f'need exactly one native ELF {soname} loader entry')
        library = compatible[0]
        if library.name != basename + RELEASE:
            raise Refusal(f'user-space library does not match loaded NVIDIA {RELEASE}: {library}')
        libraries[soname] = {'path': str(library), 'sha256': sha(library), 'elf': elf_identity(library)}
    firmware = sorted(Path('/lib/firmware/nvidia', RELEASE).glob('gsp_*.bin*'))
    if not firmware:
        raise Refusal('matching release GSP firmware artifacts were not found; refusing to infer compatibility')
    return {'release': RELEASE, 'libraries': libraries, 'ignored_incompatible_libraries': ignored,
            'firmware': {str(path.resolve()): sha(path) for path in firmware}}

def protected_path(path):
    path = Path(path).resolve(strict=True)
    for item in (path, *path.parents):
        info = item.stat()
        sticky_parent = item != path and stat.S_ISDIR(info.st_mode) and info.st_mode & stat.S_ISVTX
        if info.st_uid != 0 or (info.st_mode & 0o022 and not sticky_parent):
            raise Refusal(f'privileged kernel state/artifacts must be root-owned and not writable by others: {item}')
    return path

def protected_directory(path):
    # Check the deepest existing ancestor before creating anything below it, so
    # root never creates directories inside a location another user controls.
    path = Path(os.path.abspath(path))
    missing = []
    probe = path
    while not os.path.lexists(probe):
        missing.append(probe)
        probe = probe.parent
    protected_path(probe)
    for item in reversed(missing):
        os.mkdir(item, 0o755)
        os.chmod(item, 0o755)
    return protected_path(path)

def memory_policy():
    result = {}
    for name in ('enabled', 'defrag', 'shmem_enabled'):
        path = Path('/sys/kernel/mm/transparent_hugepage') / name
        result[name] = path.read_text().strip() if path.exists() else None
    return result

def supported():
    for name in ('nvidia_modeset', 'nvidia_drm', 'nvidia_peermem'):
        if (Path('/sys/module') / name).exists():
            raise Refusal(f'{name} is loaded; this workflow supports only nvidia and nvidia_uvm')
    for name in MODULES:
        holders = {path.name for path in (Path('/sys/module') / name / 'holders').glob('*')}
        if holders - set(MODULES):
            raise Refusal(f'{name} has other kernel clients: {", ".join(sorted(holders - set(MODULES)))}')

def idle():
    supported()
    devices = sorted(glob.glob('/dev/nvidia*'))
    devices = [path for path in devices if not Path(path).is_dir()]
    if not devices:
        raise Refusal('no NVIDIA device nodes: cannot establish an idle GPU')
    result = command(['fuser', *devices], check=False)
    if result.returncode == 0:
        raise Refusal(f'NVIDIA device users must exit first: {result.stdout.strip()} {result.stderr.strip()}')
    # fuser returns 1 for no matches AND for some errors. Any diagnostic is an
    # error, not evidence of an idle GPU. Unexpected exit codes also fail closed.
    if result.returncode != 1 or result.stdout.strip() or result.stderr.strip():
        raise Refusal(f'cannot establish idle state: fuser exit {result.returncode}: {result.stderr.strip()}')

def validate_set(directory):
    directory = Path(directory).resolve(strict=True)
    manifest = read_json(directory / 'manifest.json')
    if manifest.get('schema') != 'cmt.kernel-set' or manifest.get('version') != 1:
        raise Refusal('unsupported kernel-set manifest')
    if manifest.get('release') != RELEASE or manifest.get('variant') not in ('stock', BUNDLE, 'restore'):
        raise Refusal('kernel-set release or variant differs from the selected workflow')
    if manifest.get('kernel') != KERNEL:
        raise Refusal('kernel-set was built for a different running kernel')
    expected_modules = set(manifest.get('modules', {}))
    if ('nvidia' not in expected_modules or expected_modules - set(MODULES)
            or (manifest['variant'] != 'restore' and expected_modules != set(MODULES))):
        raise Refusal('kernel set has an invalid module inventory')
    for name in expected_modules:
        filename = MODULES[name]
        observed = module_evidence(directory / filename)
        if observed != manifest.get('modules', {}).get(name):
            raise Refusal(f'{name}: module hash/metadata differs from manifest')
        if observed['name'] != name or not observed['srcversion'] or not observed['vermagic'].split() or observed['vermagic'].split()[0] != KERNEL:
            raise Refusal(f'{name}: missing identity or incompatible vermagic')
        if observed['version'] and observed['version'] != RELEASE:
            raise Refusal(f'{name}: wrong release')
    if manifest['modules']['nvidia']['version'] != RELEASE:
        raise Refusal('nvidia module release does not match user-space')
    if manifest['variant'] in ('stock', BUNDLE) and manifest.get('source_commit') != RELEASE_ENTRY.get('source_commit'):
        raise Refusal('set does not record this release\'s exact source commit')
    if manifest['variant'] == 'stock' and manifest.get('patches') != {}:
        raise Refusal('stock set must declare no source patches')
    if manifest['variant'] == BUNDLE:
        expected = {entry['path']: sha(path) for entry, path in patch_files()}
        if manifest.get('patches') != expected:
            raise Refusal('manifest does not identify this release\'s complete A+B+D patch set')
    return directory, manifest

def parameters(directory, original):
    # This file must be supplied with known original load arguments. Do not
    # derive arguments from display strings such as '(null)' or mutable sysfs
    # attributes, and do not silently omit parameters that cannot be restored.
    evidence = read_json(directory / 'restore.json')
    if evidence.get('schema') != 'cmt.kernel-restore' or evidence.get('version') != 1:
        raise Refusal('restore.json needs schema cmt.kernel-restore, version 1')
    if evidence.get('loaded') != original:
        raise Refusal('restore.json effective parameters/identities differ from the freshly loaded state')
    arguments = evidence.get('arguments')
    if not isinstance(arguments, dict) or set(arguments) != set(MODULES):
        raise Refusal('restore.json must record original arguments for both modules')
    for name, filename in MODULES.items():
        if original[name] is None:
            if arguments[name] != []:
                raise Refusal(f'{name} was not loaded; restoration arguments must be empty')
            continue
        known = set()
        for line in command(['modinfo', '-p', directory / filename]).stdout.splitlines():
            known.add(line.split(':', 1)[0])
        seen = set()
        if not isinstance(arguments[name], list):
            raise Refusal('module arguments must be arrays')
        for value in arguments[name]:
            if not isinstance(value, str) or not replayable_argument(value) or '=' not in value:
                raise Refusal('invalid module argument')
            key, setting = value.split('=', 1)
            if key not in known or key in seen:
                raise Refusal(f'unknown or duplicate original module parameter: {key}')
            seen.add(key)
            # Require replayable exact values for every observable argument.
            observed = original[name]['parameters'].get(key)
            if observed is not None and observed != setting:
                raise Refusal(f'{key}: supplied original argument is not the observed value')
    return arguments

def replayable_argument(value):
    # insmod rebuilds a kernel command line. Its quoting cannot preserve these
    # values exactly; refuse them before unloading either module.
    return not re.search(r'''["'\x00-\x1f\x7f]''', value)

def parameter_types(path):
    result = {}
    for line in command(['modinfo', '-p', path]).stdout.splitlines():
        name, separator, description = line.partition(':')
        kind = re.search(r'\(([^()]*)\)\s*$', description)
        if not separator or not kind or name in result:
            raise Refusal(f'cannot determine replayable module parameter type: {line}')
        result[name] = kind[1]
    return result

def original_arguments(path, name, original):
    types = parameter_types(path)
    values = original['parameters']
    if set(types) != set(values):
        missing = ', '.join(sorted(set(types) - set(values)))
        raise Refusal(f'{name}: hidden/unreadable parameters ({missing}); supply --restore-set with '
                      'recorded original module arguments (see kernel_patches/nvidia-open/README.md)')
    null_defaults = RELEASE_ENTRY.get('null_default_parameters', {}).get(name, [])
    arguments = []
    for key, value in values.items():
        kind = types[key]
        if not replayable_argument(value):
            raise Refusal(f'{name}.{key}: parameter cannot be replayed as one argument')
        if kind == 'charp':
            if value == '(null)':
                if key not in null_defaults:
                    raise Refusal(f'{name}.{key}: NULL default is not source-verified; supply --restore-set')
                continue
        elif kind in ('bool', 'bint'):
            if value not in ('Y', 'N', '0', '1'):
                raise Refusal(f'{name}.{key}: ambiguous boolean parameter')
        elif kind in ('byte', 'short', 'ushort', 'int', 'uint', 'long', 'ulong', 'ullong'):
            if not re.fullmatch(r'-?(?:0x[0-9a-fA-F]+|[0-9]+)', value):
                raise Refusal(f'{name}.{key}: ambiguous integer parameter')
        else:
            raise Refusal(f'{name}.{key}: unsupported parameter type {kind}; supply --restore-set')
        arguments.append(f'{key}={value}')
    return arguments

def restoration_inputs(original, supplied):
    if supplied:
        directory, manifest = validate_set(supplied)
        matches(manifest, original)
        return directory, parameters(directory, original)
    sources, arguments = {}, {}
    for name in MODULES:
        arguments[name] = []
        if original[name] is None:
            continue
        source = Path(module_field(name, 'filename')).resolve(strict=True)
        if module_field(source, 'srcversion') != original[name]['srcversion']:
            raise Refusal(f'installed {name} differs from the loaded module; supply --restore-set')
        if note_build_id(original[name]['build_id_note']) is None:
            raise Refusal(f'loaded {name} has no build-id; automatic exact backup is unavailable')
        sources[name] = source
        arguments[name] = original_arguments(source, name, original[name])
    return sources, arguments

def restored_set(path):
    state = read_json(protected_path(path))
    if (not isinstance(state, dict) or state.get('schema') != 'cmt.kernel-state' or state.get('version') != 1
            or state.get('status') != 'restored' or state.get('boot_id') != BOOT_ID
            or state.get('kernel') != KERNEL or state.get('release') != RELEASE):
        raise Refusal('saved journal is not a restored state for this boot, kernel and NVIDIA release')
    original = loaded()
    if original != state.get('original') or original != state.get('restored'):
        raise Refusal('loaded modules or parameters differ from the saved restored state')
    if userspace() != state.get('userspace'):
        raise Refusal('user-space libraries or firmware differ from the saved restored state')
    if not isinstance(state.get('restore_set'), str):
        raise Refusal('saved journal has no original module directory')
    directory = protected_path(state['restore_set'])
    for name in ('manifest.json', 'restore.json', *(MODULES[name] for name in MODULES if original[name] is not None)):
        protected_path(directory / name)
    directory, manifest = validate_set(directory)
    if manifest != state.get('restore_manifest'):
        raise Refusal('saved original module manifest changed after restoration')
    matches(manifest, original)
    if parameters(directory, original) != state.get('original_arguments'):
        raise Refusal('recorded original module arguments changed after restoration')
    return directory

def backup_original(directory, original, supplied):
    sources, arguments = restoration_inputs(original, supplied)
    if supplied:
        sources = {name: sources / MODULES[name] for name in MODULES if original[name] is not None}
    directory.mkdir()
    modules = {}
    for name, source in sources.items():
        destination = directory / MODULES[name]
        if source.suffix in ('.gz', '.xz'):
            opener = gzip.open if source.suffix == '.gz' else lzma.open
            with opener(source, 'rb') as incoming, open(destination, 'wb') as outgoing:
                shutil.copyfileobj(incoming, outgoing)
        elif source.suffix == '.zst':
            with open(destination, 'wb') as outgoing:
                result = subprocess.run(['zstd', '-dc', str(source)], stdout=outgoing, stderr=subprocess.PIPE)
            if result.returncode:
                raise Refusal(f'cannot decompress original {name}: {result.stderr.decode(errors="replace")}')
        else:
            shutil.copy2(source, destination)
        destination.chmod(0o644)
        modules[name] = module_evidence(destination)
        known_id = note_build_id(original[name]['build_id_note'])
        if known_id is not None and modules[name]['build_id'] != known_id:
            raise Refusal(f'{name}: backup build-id differs from the loaded module')
    manifest = {'schema': 'cmt.kernel-set', 'version': 1, 'release': RELEASE, 'variant': 'restore',
                'kernel': KERNEL, 'modules': modules}
    atomic_json(directory / 'manifest.json', manifest)
    atomic_json(directory / 'restore.json', {'schema': 'cmt.kernel-restore', 'version': 1,
                                           'loaded': original, 'arguments': arguments})
    validate_set(directory)
    matches(manifest, original)
    for path in directory.iterdir():
        protected_path(path)
    return manifest, arguments

def matches(manifest, actual):
    for name in MODULES:
        expected = manifest['modules'].get(name)
        if expected is None and actual.get(name) is None:
            continue
        if expected is None or actual.get(name) is None or actual[name]['srcversion'] != expected['srcversion']:
            raise Refusal(f'loaded {name} does not match the declared set')
        known_id = note_build_id(actual[name].get('build_id_note'))
        if expected.get('build_id') is not None and expected['build_id'] != known_id:
            raise Refusal(f'loaded {name} build-id differs from the declared set')

def unload():
    for name in ('nvidia_uvm', 'nvidia'):
        if (Path('/sys/module') / name).exists():
            command(['rmmod', name])
        if (Path('/sys/module') / name).exists():
            raise Refusal(f'{name} is still loaded after rmmod')

def load(directory, arguments, manifest):
    protected_path(directory)
    protected_path(directory / 'manifest.json')
    for name in manifest['modules']:
        protected_path(directory / MODULES[name])
    if validate_set(directory)[1] != manifest:
        raise Refusal('module files changed immediately before loading')
    unload()
    for name, filename in MODULES.items():
        if name not in manifest['modules']:
            continue
        protected_path(directory / filename)
        if module_evidence(directory / filename) != manifest['modules'][name]:
            raise Refusal(f'{name}: artifact changed before insmod')
        command(['insmod', directory / filename, *arguments[name]])
    # Never invoke nvidia-modprobe: installed modules may differ from this set.
    majors = {}
    for line in Path('/proc/devices').read_text().splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0].isdigit():
            majors[fields[1]] = int(fields[0])
    for path, minor in (('/dev/nvidia-uvm', 0), ('/dev/nvidia-uvm-tools', 1)):
        if 'nvidia_uvm' not in manifest['modules']:
            break
        node = Path(path).stat()
        if (not stat.S_ISCHR(node.st_mode) or os.major(node.st_rdev) != majors.get('nvidia-uvm')
                or os.minor(node.st_rdev) != minor):
            raise Refusal(f'{path}: retained device node does not match selected UVM; explicit node recovery required')
    matches(manifest, loaded())
    command(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'])

def journal(state, path, status, **fields):
    state.update(fields)
    state['status'] = status
    state.setdefault('events', []).append({'time_unix': time.time(), 'status': status})
    atomic_json(path, state)

def report_failure(state, path, status, error):
    try:
        journal(state, path, status, failure=str(error))
    except OSError as write_error:
        print(f'Could not update recovery journal {path}: {write_error}', file=sys.stderr)

# Where the default measurement lock lives (src/runner.h, MeasurementLockPlaces). Fixed; tests point them at
# scratch directories.
RUNTIME_BASE = '/run/user'
FALLBACK_BASE = '/tmp'
LOCK_NAME = 'cmt-measurements.lock'

def measurement_uid():
    # Under sudo the measurement lock belongs to the invoking user, not root.
    sudo_uid = os.environ.get('SUDO_UID', '')
    return int(sudo_uid) if os.geteuid() == 0 and sudo_uid.isdigit() else os.geteuid()

def fallback_lock_directory(uid):
    return os.path.join(FALLBACK_BASE, f'cmt-{uid}')

def default_measurement_lock():
    """The lock `cmt tune` and cmt-evaluate hold while measuring (src/runner.cpp, MeasurementLock):
    /run/user/<uid>/cmt-measurements.lock where that per-user runtime directory exists, else
    /tmp/cmt-<uid>/cmt-measurements.lock in a private directory; under sudo, uid is the invoking user's."""
    inherited = os.environ.get('CMT_MEASUREMENT_LOCK_PATH')
    if inherited:
        return os.path.abspath(inherited)
    uid = measurement_uid()
    runtime = os.path.join(RUNTIME_BASE, str(uid))
    try:
        os.lstat(runtime)
    except FileNotFoundError:
        return os.path.join(fallback_lock_directory(uid), LOCK_NAME)
    except OSError:
        pass  # present but not inspectable: measurement_owner reports it
    return os.path.join(runtime, LOCK_NAME)

def private_lock_directory(uid):
    """/tmp/cmt-<uid> as an open descriptor: created 0700 if missing (given to uid when root creates it under
    sudo), opened without following a symlink, and refused unless it is a directory owned by uid with mode exactly
    0700. Anyone can create a name in /tmp first; these checks make a squatter fail loudly."""
    path = fallback_lock_directory(uid)
    name = os.path.basename(path)
    try:
        base = os.open(FALLBACK_BASE, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    except OSError as error:
        raise Refusal(f'cannot open {FALLBACK_BASE} for the measurement lock directory: {error}') from None
    try:
        created = False
        try:
            os.mkdir(name, 0o700, dir_fd=base)
            created = True
        except FileExistsError:
            pass
        try:
            directory = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=base)
        except OSError as error:
            raise Refusal(f'measurement lock directory {path} is not a directory, or is a symlink (not followed): '
                          f'{error}; inspect and remove it, or pass --measurement-lock') from None
    except OSError as error:
        raise Refusal(f'cannot create measurement lock directory {path}: {error}') from None
    finally:
        os.close(base)
    try:
        if created:
            os.fchmod(directory, 0o700)  # mkdir's mode is masked by the umask
            if os.geteuid() == 0 and uid != 0:
                os.fchown(directory, uid, -1)
        info = os.fstat(directory)
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != uid or stat.S_IMODE(info.st_mode) != 0o700:
            raise Refusal(f'measurement lock directory {path} must be a directory owned by uid {uid} with mode 0700 '
                          f'(it is owned by uid {info.st_uid}, mode {stat.S_IMODE(info.st_mode):04o}); inspect and '
                          'remove it, or pass --measurement-lock')
    except BaseException:
        os.close(directory)
        raise
    return directory

def measurement_owner(path):
    # Never resolve symlinks here, or create directories other than the private
    # fallback one: the lock usually lives in the invoking user's runtime
    # directory, which root must not trust.
    path = os.path.abspath(path)
    parent, name = os.path.split(path)
    uid = measurement_uid()
    if os.path.normpath(parent) == os.path.normpath(fallback_lock_directory(uid)):
        directory = private_lock_directory(uid)
    else:
        try:
            directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        except FileNotFoundError:
            raise Refusal(f'measurement lock directory {parent} does not exist; pass --measurement-lock (or set '
                          'CMT_MEASUREMENT_LOCK_PATH) to the lock your cmt tune sessions use') from None
        except NotADirectoryError:
            raise Refusal(f'measurement lock directory {parent} is not a directory, or is a symlink '
                          '(not followed)') from None
        except OSError as error:
            raise Refusal(f'cannot open measurement lock directory {parent} without following symlinks: '
                          f'{error}') from None
    try:
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
        created = False
        try:
            descriptor = os.open(name, flags, dir_fd=directory)
        except FileNotFoundError:
            try:
                # Created as cmt creates it (0600).
                descriptor = os.open(name, flags | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=directory)
                created = True
            except FileExistsError:
                descriptor = os.open(name, flags, dir_fd=directory)
    except OSError as error:
        raise Refusal(f'cannot open measurement lock {path} without following symlinks: {error}') from None
    finally:
        os.close(directory)
    lock = os.fdopen(descriptor, 'rb')
    try:
        if created and os.geteuid() == 0 and uid != 0:
            # A root-owned lock would make the user's next `cmt tune` fail to open its own lock.
            os.fchown(lock.fileno(), uid, -1)
        info = os.fstat(lock.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise Refusal('measurement lock must name a regular lock file')
        if info.st_uid not in (uid, 0):
            raise Refusal(f'measurement lock {path} is owned by uid {info.st_uid}, not by uid {uid} or root')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Refusal(f'another measurement owns {path}') from None
    except BaseException:
        lock.close()
        raise
    return lock

def root_directory(parent, name):
    # Open (creating if needed) a root-owned 0700 directory without following symlinks.
    base = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        created = False
        try:
            os.mkdir(name, 0o700, dir_fd=base)
            created = True
        except FileExistsError:
            pass
        try:
            directory = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=base)
        except OSError as error:
            raise Refusal(f'{os.path.join(parent, name)} must be a root-owned 0700 directory: {error}') from None
    finally:
        os.close(base)
    if created:
        os.fchmod(directory, 0o700)
    info = os.fstat(directory)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != 0 or stat.S_IMODE(info.st_mode) != 0o700:
        os.close(directory)
        raise Refusal(f'{os.path.join(parent, name)} must be a root-owned 0700 directory; '
                      'inspect and remove it, then retry')
    return directory

def mutation_owner():
    if os.geteuid() != 0:
        raise Refusal('loading/restoring kernel modules requires root; invoke this explicit command with sudo')
    if os.environ.get('LD_PRELOAD') or os.environ.get('LD_LIBRARY_PATH'):
        raise Refusal('kernel run/restore requires the system loader; unset LD_PRELOAD and LD_LIBRARY_PATH')
    directory = root_directory(MUTATION_LOCK_PARENT, MUTATION_LOCK_DIRECTORY)
    try:
        descriptor = os.open(MUTATION_LOCK_NAME, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600,
                             dir_fd=directory)
    finally:
        os.close(directory)
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
        os.close(descriptor)
        raise Refusal('kernel mutation lock must be a root-owned regular file, not writable by others')
    lock = os.fdopen(descriptor, 'r+b')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise Refusal('another kernel operation owns this machine')
    return lock

def load_prerequisites():
    require_minimum_kernel()
    if Path('/proc/sys/kernel/modules_disabled').read_text().strip() != '0':
        raise Refusal('kernel module loading is disabled until reboot')
    capabilities = re.search(r'^CapEff:\s*([0-9a-fA-F]+)$', Path('/proc/self/status').read_text(), re.M)
    if os.geteuid() != 0 or capabilities is None or not int(capabilities[1], 16) & (1 << CAP_SYS_MODULE):
        raise Refusal('root with CAP_SYS_MODULE is required; this process cannot load modules')
    lockdown = Path('/sys/kernel/security/lockdown')
    if lockdown.exists() and '[none]' not in lockdown.read_text():
        raise Refusal('kernel lockdown requires trusted signed modules; this workflow does not sign modules')
    enforce = Path('/proc/sys/kernel/module_sig_enforce')
    if enforce.exists() and enforce.read_text().strip() == '1':
        raise Refusal('kernel requires signed modules; this workflow does not sign modules')
    configurations = []
    for path in (Path('/boot') / f'config-{KERNEL}', Path('/lib/modules') / KERNEL / 'build/include/config/auto.conf'):
        if path.is_file():
            configurations.append(path.read_text())
        if path.is_file() and re.search(r'^CONFIG_MODULE_SIG_FORCE=y$', configurations[-1], re.M):
            raise Refusal('CONFIG_MODULE_SIG_FORCE requires a separate trusted signing workflow')
    if not any(re.search(r'^CONFIG_MODULE_UNLOAD=y$', config, re.M) for config in configurations):
        raise Refusal('cannot establish CONFIG_MODULE_UNLOAD=y for the running kernel')
    driver = Path('/proc/driver/nvidia/version').read_text()
    if 'Open Kernel Module' not in driver:
        raise Refusal('this workflow requires an already working matching NVIDIA open module; '
                      'open-module support is not established for the current GPU/driver')
    if RELEASE not in driver:
        raise Refusal('loaded NVIDIA release differs between sysfs and procfs')
    for name in ('nvidia-uvm', 'nvidia-uvm-tools'):
        if not (Path('/dev') / name).is_char_device():
            raise Refusal(f'/dev/{name} is absent or not a character device; establish UVM device nodes before patching')
    if os.environ.get('LD_PRELOAD') or os.environ.get('LD_LIBRARY_PATH'):
        raise Refusal('unset LD_PRELOAD and LD_LIBRARY_PATH before changing kernel modules')
    for tool in ('modinfo', 'insmod', 'rmmod', 'ldconfig', 'fuser', 'readelf', 'nvidia-smi'):
        if shutil.which(tool) is None:
            raise Refusal(f'required module tool is unavailable: {tool}')

def build_prerequisites():
    patch_files()
    require_minimum_kernel()
    for tool in ('git', 'make', 'cc', 'c++', 'ld', 'readelf', 'modinfo'):
        if shutil.which(tool) is None:
            raise Refusal(f'required build tool is unavailable: {tool}')
    headers = Path('/lib/modules') / KERNEL / 'build'
    if not (headers / 'Makefile').is_file() or not (headers / 'Module.symvers').is_file():
        raise Refusal(f'complete headers and Module.symvers for running kernel {KERNEL} are required')
    exported = (headers / 'Module.symvers').read_text()
    for symbol in ('pin_user_pages_fast', 'unpin_user_pages_dirty_lock'):
        if not re.search(r'\s' + symbol + r'\s', exported):
            raise Refusal(f'running kernel headers do not export required API {symbol}')
    return str(headers.resolve())

def build_identity(headers):
    headers = Path(headers)
    inputs = ('Module.symvers', 'include/config/auto.conf', 'include/generated/utsrelease.h')
    return {'kernel_build': Path('/proc/version').read_text().strip(),
            'headers': str(headers), 'header_sha256': {name: sha(headers / name) for name in inputs},
            'compiler': {name: {'path': str(Path(shutil.which(name)).resolve(strict=True)),
                                'version': command([name, '--version']).stdout}
                         for name in ('cc', 'c++', 'ld')}}

def preflight(args):
    evidence = {'schema': 'cmt.kernel-check', 'version': 1, 'bundle': BUNDLE,
                'release': RELEASE, 'kernel': KERNEL, 'boot_id': BOOT_ID,
                'source_status': RELEASE_ENTRY.get('status'), 'checks': {}}
    evidence['restoration_source'] = 'explicit' if args.restore_set else 'automatic_backup'
    if not args.restore_set and Path(args.state).exists():
        try:
            args.restore_set = str(restored_set(args.state))
            evidence['restoration_source'] = 'restored_journal'
            evidence['restoration_journal'] = str(Path(args.state).resolve())
            evidence['restore_set'] = args.restore_set
        except (Refusal, OSError, ValueError, KeyError, TypeError) as error:
            evidence['restored_journal_not_reused'] = str(error)
    try:
        original = loaded()
        evidence['loaded'] = original
        evidence['original_artifacts'] = {}
        for name, current in original.items():
            if current is None:
                continue
            candidate = (Path(args.restore_set) / MODULES[name]) if args.restore_set else Path(module_field(name, 'filename'))
            candidate = candidate.resolve(strict=True)
            compressed = candidate.suffix in ('.xz', '.gz', '.zst')
            evidence['original_artifacts'][name] = {
                'path': str(candidate), 'srcversion': module_field(candidate, 'srcversion'),
                'build_id': None if compressed else module_build_id(candidate),
                'build_id_note_loaded': current['build_id_note'],
                'hidden_parameters': sorted(set(parameter_types(candidate)) - set(current['parameters'])),
                'identity_limit': 'compressed build-id checked after backup, before unload' if compressed else None,
            }
    except (Refusal, OSError, ValueError) as error:
        evidence['original_artifact_error'] = str(error)
    checks = [('build', lambda: build_identity(build_prerequisites())), ('load_permission', load_prerequisites),
              ('userspace', userspace), ('idle', idle),
              ('restoration', lambda: restoration_inputs(loaded(), args.restore_set))]
    for name, operation in checks:
        try:
            operation()
            evidence['checks'][name] = {'available': True}
        except (Refusal, OSError, ValueError) as error:
            evidence['checks'][name] = {'available': False, 'reason': str(error)}
    evidence['can_build'] = evidence['checks']['build']['available']
    evidence['can_attempt_load'] = all(item['available'] for item in evidence['checks'].values())
    evidence['limits'] = 'Preflight does not prove compilation, successful loading, or 2 MiB GPU mappings.'
    return evidence

def restore_state(state, path):
    if state.get('boot_id') != BOOT_ID or state.get('kernel') != KERNEL:
        raise Refusal('restoration journal belongs to another boot or kernel; no modules were changed')
    directory, manifest = validate_set(state['restore_set'])
    if manifest != state['restore_manifest']:
        raise Refusal('restoration manifest changed after the original snapshot')
    if parameters(directory, state['original']) != state['original_arguments']:
        raise Refusal('original module argument evidence changed since the journal was captured')
    if userspace() != state['userspace']:
        raise Refusal('user-space library changed since the original snapshot')
    if loaded() == state['original']:
        journal(state, path, 'restored', restored=state['original'], restored_memory_policy=memory_policy())
        return
    supported()
    # Device nodes can disappear after an interrupted unload. If nvidia remains
    # loaded, require a clean idle check; otherwise there is no loaded CUDA owner.
    if (Path('/sys/module/nvidia')).exists():
        idle()
    try:
        journal(state, path, 'restoring')
    except OSError as error:
        print(f'Could not record restoration start in {path}: {error}; restoring from saved evidence.', file=sys.stderr)
    load(directory, state['original_arguments'], manifest)
    actual = loaded()
    if actual != state['original']:
        raise Refusal('original module identities or effective parameters were not restored exactly')
    journal(state, path, 'restored', restored=actual, restored_memory_policy=memory_policy())

def build(args):
    headers = build_prerequisites()
    identity = build_identity(headers)
    patch_list = patch_files()
    commit = RELEASE_ENTRY['source_commit']
    output = Path(args.out).resolve()
    if output.exists():
        raise Refusal('build output must be a new directory; existing artifacts are preserved')
    output.mkdir(parents=True)
    source = output / 'sources' / 'pristine'
    source.parent.mkdir()
    if args.source:
        # A local checkout supplies Git objects only. Never copy its working
        # changes or edit its worktree; the catalog commit selects the source.
        # Objects are fetched (and hash-checked) rather than shared, so later
        # changes to the checkout cannot alter this build. Git reads the
        # checkout's configuration, which can run commands: as root, accept
        # only a checkout that root owns and nobody else can write.
        local = Path(args.source).resolve(strict=True)
        if os.geteuid() == 0:
            protected_path(local)
            for name in ('.git', '.git/config', 'config'):
                if os.path.lexists(local / name):
                    protected_path(local / name)
        command(['git', 'init', source])
        command(['git', 'fetch', '--depth=1', local, commit], cwd=source)
    else:
        command(['git', 'init', source])
        print(f'Fetching NVIDIA {RELEASE}, exact source commit {commit}', flush=True)
        command(['git', 'fetch', '--depth=1', SOURCE_URL, commit], cwd=source)
    command(['git', 'checkout', '--detach', commit], cwd=source)
    if command(['git', 'rev-parse', 'HEAD'], cwd=source).stdout.strip() != commit:
        raise Refusal('source checkout does not match the catalog commit')
    for variant in ('stock', BUNDLE):
        tree = output / 'sources' / variant
        command(['git', 'worktree', 'add', '--detach', tree, commit], cwd=source)
        patches = {}
        if variant == BUNDLE:
            for entry, path in patch_list:
                prefix = ['--directory=' + entry['directory']] if entry.get('directory') else []
                command(['git', 'apply', '--check', *prefix, path], cwd=tree)
                command(['git', 'apply', *prefix, path], cwd=tree)
                patches[entry['path']] = sha(path)
        destination = output / variant
        destination.mkdir()
        # Top-level modules rebuilds RM source and the interface together.
        print(f'Building {variant}; log: {destination / "build.log"}', flush=True)
        with open(destination / 'build.log', 'w') as log:
            result = subprocess.run(['make', f'-j{args.jobs}', 'modules',
                                     'CC=' + identity['compiler']['cc']['path'],
                                     'CXX=' + identity['compiler']['c++']['path'],
                                     'LD=' + identity['compiler']['ld']['path'],
                                     'NV_EXCLUDE_KERNEL_MODULES=nvidia-drm nvidia-modeset nvidia-peermem'],
                                    cwd=tree, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise Refusal(f'{variant} build failed; see {destination}/build.log')
        for filename in MODULES.values():
            shutil.copy2(tree / 'kernel-open' / filename, destination / filename)
        manifest = {'schema': 'cmt.kernel-set', 'version': 1, 'release': RELEASE, 'variant': variant,
                    'kernel': KERNEL, 'source_commit': commit, 'source_path': str(tree), 'patches': patches,
                    'catalog_sha256': sha(SCRIPT_DIR / 'releases.json'),
                    'build_identity': identity,
                    'verification': {'patches_applied': True, 'built': True, 'loaded': False},
                    'compiler': command(['cc', '--version']).stdout,
                    'kernel_headers': headers,
                    'modules': {name: module_evidence(destination / filename) for name, filename in MODULES.items()}}
        atomic_json(destination / 'manifest.json', manifest)
        validate_set(destination)
    print(f'Built stock and {BUNDLE} sets in {output}; no loaded modules were changed.')

def prepare_state(path, target, manifest, restore, restore_manifest, lock_path, original, arguments):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    os.close(descriptor)
    state = {'schema': 'cmt.kernel-state', 'version': 1, 'kernel': KERNEL,
             'boot_id': BOOT_ID, 'release': RELEASE, 'bundle': BUNDLE,
             'measurement_lock': os.path.abspath(lock_path),
             'restore_set': str(restore), 'restore_manifest': restore_manifest,
             'requested_set': str(target), 'requested_manifest': manifest,
             'original': original, 'original_arguments': arguments, 'userspace': userspace(),
             'original_memory_policy': memory_policy()}
    journal(state, path, 'prepared')
    return state

def switch_set(state, path, large_system_pages):
    load_prerequisites()
    idle()
    if loaded() != state['original'] or detect_release() != RELEASE or userspace() != state['userspace']:
        raise Refusal('module or user-space state changed after preparation; no modules were unloaded')
    target, manifest = validate_set(state['requested_set'])
    if manifest != state['requested_manifest']:
        raise Refusal('selected module manifest changed after preparation')
    arguments = {name: list(values) for name, values in state['original_arguments'].items()}
    expected = {name: dict(value['parameters']) if value is not None else None
                for name, value in state['original'].items()}
    if large_system_pages:
        if manifest['variant'] != BUNDLE:
            raise Refusal('large system pages require the complete host-registration patch set')
        key = 'NVreg_RegistryDwords'
        previous = expected['nvidia'].get(key)
        if previous is None:
            raise Refusal('cannot observe NVreg_RegistryDwords for the D opt-in')
        fields = [] if previous == '(null)' else [part for part in previous.split(';') if part.strip()]
        fields = [part for part in fields if part.split('=', 1)[0].strip() != 'RMAllowSysmemLargePages']
        fields.append('RMAllowSysmemLargePages=1')
        setting = ';'.join(fields)
        arguments['nvidia'] = [value for value in arguments['nvidia'] if not value.startswith(key + '=')]
        arguments['nvidia'].append(f'{key}={setting}')
        expected['nvidia'][key] = setting
    journal(state, path, 'switching', selected_arguments=arguments,
            large_system_pages='requested' if large_system_pages else 'not_requested')
    load(target, arguments, manifest)
    observed = loaded()
    for name in MODULES:
        if expected[name] is not None and observed[name]['parameters'] != expected[name]:
            raise Refusal(f'{name}: loaded parameters differ from the declared settings')
    journal(state, path, 'active', observed=observed, observed_memory_policy=memory_policy(),
            mapping_evidence='not_measured')

def rollback(state, path):
    # Children started from here (rmmod, insmod) inherit these ignored signals.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        restore_state(state, path)
    except BaseException as error:
        report_failure(state, path, 'restore_failed', error)
        print(f'RESTORATION FAILED. Preserve {path}; use cmt unpatch --state {path}. {error}', file=sys.stderr)
        raise

def prepared_target(args):
    source_identity = {'commit': RELEASE_ENTRY['source_commit'], 'patches': RELEASE_ENTRY['patches'],
                       'build': build_identity(build_prerequisites())}
    digest = hashlib.sha256(json.dumps(source_identity, sort_keys=True).encode()).hexdigest()[:12]
    key = f'{RELEASE}-{KERNEL}-{os.uname().machine}-{digest}'
    output = Path(args.work_dir).resolve() / key
    if not output.exists():
        args.out = str(output)
        build(args)
    else:
        validate_set(output / 'stock')
    return validate_set(output / BUNDLE)

def patch(args):
    evidence = preflight(args)
    if args.check:
        print(json.dumps(evidence, indent=2, sort_keys=True))
        return 0 if evidence['can_attempt_load'] else 2
    required = ('build',) if args.build_only else tuple(evidence['checks'])
    errors = [evidence['checks'][name]['reason'] for name in required
              if not evidence['checks'][name]['available']]
    if errors:
        raise Refusal('; '.join(errors))
    with measurement_owner(args.measurement_lock):
        with contextlib.nullcontext() if args.build_only else mutation_owner():
            if not args.build_only or os.geteuid() == 0:
                # Root checks the location before creating anything in it.
                protected_directory(args.work_dir)
            target, manifest = prepared_target(args)
            if args.build_only:
                print(f'Ready: {target}. Modules remain unchanged.')
                return 0
            path = Path(args.state).resolve()
            protected_directory(path.parent)
            if path.exists() and existing_journal(path) == 'active':
                print(f'{BUNDLE} is already active; restore with cmt unpatch --state {path}')
                return 0
            idle()
            original = loaded()
            restore = path.parent / f'original-{time.time_ns()}'
            restore_manifest, arguments = backup_original(restore, original, args.restore_set)
            state = prepare_state(path, target, manifest, restore, restore_manifest,
                                  args.measurement_lock, original, arguments)
            switch_or_rollback(state, path)
            print(f'{BUNDLE} is active for NVIDIA {RELEASE}; A+B+D installed, D mapping size not measured.')
            print(f'Restore the exact previous modules and parameters: cmt unpatch --state {path}')
    return 0

def switch_or_rollback(state, path):
    try:
        switch_set(state, path, True)
    except BaseException as error:
        # Once the 'switching' record exists, modules may already be unloaded.
        needs_restore = 'selected_arguments' in state
        report_failure(state, path, 'patch_failed', error)
        if needs_restore:
            rollback(state, path)
        raise

def existing_journal(path):
    """What an existing journal means for a new patch: 'active' when the bundle
    is already active in this boot; None after archiving a finished or stale
    journal. Anything else is refused."""
    previous = read_json(path)
    if not isinstance(previous, dict) or previous.get('schema') != 'cmt.kernel-state':
        raise Refusal(f'{path} is not a cmt kernel journal; inspect it and move it aside')
    if previous.get('status') == 'restored':
        history = path.parent / f'{path.stem}-restored-{time.time_ns()}.json'
        os.replace(path, history)
        return None
    if previous.get('boot_id') != BOOT_ID:
        archive_stale_journal(path, previous)
        return None
    if previous.get('status') == 'active':
        if loaded() != previous.get('observed'):
            raise Refusal('an active journal exists but loaded state changed; preserve it and use unpatch')
        return 'active'
    raise Refusal(f'an unfinished recovery journal exists: {path}; use unpatch before patching')

def archive_stale_journal(path, state):
    # A journal from another boot cannot be replayed: the reboot discarded the
    # module switch it records. If exactly the original modules and parameters
    # it recorded are loaded again, nothing is left to restore.
    if loaded() != state.get('original'):
        raise Refusal(
            f'{path} (status {state.get("status")!r}) was written during another boot ({state.get("boot_id")}), '
            'and the loaded NVIDIA modules or parameters differ from the original ones it recorded, so it cannot '
            'be replayed or retired automatically. Check the loaded modules with `cmt kernel inspect` and load '
            'the driver you want (for example, reboot into the original configuration); the original modules '
            f'it saved are in {state.get("restore_set")}. Then, as root, move the journal aside '
            f'(mv {path} {path.parent}/{path.stem}-stale-manual.json) and retry.')
    history = path.parent / f'{path.stem}-stale-{time.time_ns()}.json'
    os.replace(path, history)
    print(f'{path} was left by a previous boot (status {state.get("status")!r}); the original modules and '
          f'parameters it recorded are loaded again, so it was archived as {history}.', file=sys.stderr)
    return history

def run(args):
    with measurement_owner(args.measurement_lock) as measurement, mutation_owner():
        load_prerequisites()
        target, manifest = validate_set(args.set)
        if manifest['variant'] not in ('stock', BUNDLE):
            raise Refusal('run target must be a built stock or host-registration set')
        restore, restore_manifest = validate_set(args.restore_set)
        protected_path(target)
        protected_path(restore)
        protected_path(restore / 'restore.json')
        protected_path(restore / 'manifest.json')
        for name in restore_manifest['modules']:
            protected_path(restore / MODULES[name])
        path = Path(args.state).resolve()
        if not path.parent.is_dir():
            raise Refusal('state parent directory must already exist')
        protected_path(path.parent)
        idle()
        original = loaded()
        matches(restore_manifest, original)
        arguments = parameters(restore, original)
        state = prepare_state(path, target, manifest, restore, restore_manifest,
                              args.measurement_lock, original, arguments)
        state['command'] = args.command
        switched = False
        process = None
        pending_signal = None
        outcome = 1
        try:
            switch_set(state, path, args.large_system_pages)
            switched = True
            environment = dict(os.environ)
            environment['CMT_MEASUREMENT_LOCK_FD'] = str(measurement.fileno())
            environment['CMT_MEASUREMENT_LOCK_PATH'] = state['measurement_lock']
            def forward(number, frame):
                nonlocal pending_signal
                if process is None:
                    pending_signal = number
                else:
                    try:
                        os.killpg(process.pid, number)
                    except ProcessLookupError:
                        pass
            previous = {number: signal.signal(number, forward) for number in forwarded_signals()}
            try:
                process = subprocess.Popen(args.command, start_new_session=True, env=environment,
                                           pass_fds=(measurement.fileno(),), cwd=ORIGINAL_CWD)
                if pending_signal is not None:
                    forward(pending_signal, None)
                outcome = process.wait()
            finally:
                for number, handler in previous.items():
                    signal.signal(number, handler)
            journal(state, path, 'command_finished', command_exit=outcome)
        except BaseException as error:
            # Stop only the process group created above before restoring CUDA.
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGHUP, signal.SIG_IGN)
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                except ProcessLookupError:
                    process.wait()
            switched = switched or 'selected_arguments' in state
            outcome = 1
            report_failure(state, path, 'run_failed', error)
            print(f'kernel run failed: {error}', file=sys.stderr)
        finally:
            if switched:
                try:
                    rollback(state, path)
                except BaseException:
                    outcome = 2
        return outcome if outcome >= 0 else 128 - outcome

def unpatch(args):
    path = protected_path(args.state)
    before_lock = read_json(path)
    with measurement_owner(args.measurement_lock or before_lock['measurement_lock']), mutation_owner():
        protected_path(path)
        state = read_json(path)
        if state != before_lock:
            raise Refusal('recovery journal changed while acquiring measurement ownership')
        if state.get('schema') != 'cmt.kernel-state' or state.get('version') != 1:
            raise Refusal('unsupported recovery journal')
        if state.get('boot_id') != BOOT_ID:
            archive_stale_journal(path, state)
            print('Nothing to restore: the original modules and parameters are loaded.')
            return 0
        if state.get('kernel') != KERNEL:
            raise Refusal('journal records a different running kernel')
        if state.get('status') == 'restored':
            if loaded() != state['original']:
                raise Refusal('journal says restored, but machine changed; refusing a new mutation')
            print('Original modules and parameters remain restored.')
            return 0
        if 'original_arguments' not in state:
            raise Refusal('journal contains no validated restoration snapshot; recovery is unavailable')
        rollback(state, path)
    print('Restored the exact previous modules and observed parameters.')
    return 0

def main():
    parser = argparse.ArgumentParser(prog='cmt kernel',
        description='Explicit host-memory registration optimizations. Tune never changes modules.',
        epilog='A source-applicable patch is not yet a compiled or loaded module. '
               'Builds do not prove performance or large GPU mappings. Keep the state journal for unpatch.')
    actions = parser.add_subparsers(dest='action', required=True)
    actions.add_parser('inspect', help='Read loaded module, parameter and user-space identity')
    apply = actions.add_parser('patch', prog='cmt patch', help='Check, fetch and build matching A+B+D; preserve originals and load')
    mode = apply.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true', help='Read-only preflight; no files, downloads or module changes')
    mode.add_argument('--build-only', action='store_true', help='Fetch/apply/build; leave loaded modules unchanged')
    apply.add_argument('--work-dir', help='Source and build artifact cache (default: /var/cache/cmt/kernel '
                                          'as root, otherwise ~/.cache/cmt/kernel)')
    apply.add_argument('--state', default=DEFAULT_STATE, help='Persistent recovery journal')
    apply.add_argument('--restore-set', help='Recorded original .ko files, manifest.json and restore.json when auto-backup is ambiguous')
    apply.add_argument('--source', help='Optional local Git object source; its working tree remains unchanged')
    lock_help = ('The lock cmt tune holds while measuring (default: CMT_MEASUREMENT_LOCK_PATH, else '
                 '/run/user/UID/cmt-measurements.lock, or /tmp/cmt-UID/cmt-measurements.lock without '
                 "/run/user/UID; under sudo, UID is the invoking user's)")
    apply.add_argument('--measurement-lock', help=lock_help)
    apply.add_argument('--jobs', type=int, default=4)
    undo = actions.add_parser('unpatch', prog='cmt unpatch', help='Restore exactly the modules and parameters saved by patch')
    undo.add_argument('--state', default=DEFAULT_STATE)
    undo.add_argument('--measurement-lock', help='Measurement lock to hold (default: the one recorded in the journal)')
    create = actions.add_parser('build', help='Build stock and host-registration sets for the loaded release')
    create.add_argument('--measurement-lock', help=lock_help)
    create.add_argument('--source', help='Optional local Git object source')
    create.add_argument('--out', required=True)
    create.add_argument('--jobs', type=int, default=4)
    measure = actions.add_parser('run', help='Load a built set, run one command, then restore')
    measure.add_argument('--measurement-lock', required=True)
    measure.add_argument('--set', required=True)
    measure.add_argument('--restore-set', required=True)
    measure.add_argument('--state', required=True, help='Fresh recovery journal; existing evidence is preserved')
    measure.add_argument('--large-system-pages', action='store_true', help='Request D mappings for an explicit host-registration run')
    measure.add_argument('command', nargs=argparse.REMAINDER)
    recover = actions.add_parser('restore', help='Restore an interrupted explicit run')
    recover.add_argument('--state', required=True)
    recover.add_argument('--measurement-lock', help='Measurement lock to hold (default: the one recorded in the journal)')
    args = parser.parse_args()
    if hasattr(args, 'jobs') and not 1 <= args.jobs <= 256:
        raise Refusal('--jobs must be 1..256')
    prepare_arguments(args)
    if args.action in ('restore', 'unpatch'):
        select_journal_release(read_json(protected_path(args.state)))
    else:
        select_release(detect_release())
    if args.action == 'inspect':
        evidence = {'kernel': KERNEL, 'boot_id': BOOT_ID, 'release': RELEASE,
                    'loaded': loaded(), 'memory_policy': memory_policy()}
        for name, operation in (('idle', idle), ('userspace', userspace)):
            try:
                evidence[name] = operation() if name == 'userspace' else operation() is None
            except (Refusal, OSError) as error:
                evidence[name + '_error'] = str(error)
        print(json.dumps(evidence, indent=2, sort_keys=True))
    elif args.action == 'patch':
        return patch(args)
    elif args.action == 'build':
        with measurement_owner(args.measurement_lock):
            build(args)
    elif args.action == 'run':
        if args.command[:1] != ['--'] or len(args.command) < 2:
            raise Refusal('run requires -- followed by the application command')
        args.command = args.command[1:]
        return run(args)
    else:
        return unpatch(args)
    return 0

def default_work_dir():
    # Root never defaults to a cache in a user's home directory.
    return '/var/cache/cmt/kernel' if os.geteuid() == 0 else str(Path.home() / '.cache/cmt/kernel')

def prepare_arguments(args):
    # Make user-supplied paths absolute (without resolving symlinks), then leave
    # the caller's directory so no tool runs in, or reads from, a directory
    # another user may control.
    global ORIGINAL_CWD
    ORIGINAL_CWD = os.getcwd()
    if args.action == 'patch' and args.work_dir is None:
        args.work_dir = default_work_dir()
    if args.action in ('patch', 'build') and args.measurement_lock is None:
        args.measurement_lock = default_measurement_lock()
    for name in ('work_dir', 'state', 'restore_set', 'source', 'measurement_lock', 'out', 'set'):
        value = getattr(args, name, None)
        if value:
            setattr(args, name, os.path.abspath(value))
    os.chdir('/')

def interrupted(number, frame):
    raise InterruptedError(f'interrupted by signal {number}')

def forwarded_signals():
    # A hangup that the caller already ignores (e.g. under nohup) stays ignored.
    hangup = () if signal.getsignal(signal.SIGHUP) == signal.SIG_IGN else (signal.SIGHUP,)
    return (signal.SIGINT, signal.SIGTERM, *hangup)

if __name__ == '__main__':
    # A hangup (e.g. a dropped SSH session) must roll back like SIGTERM rather
    # than kill the interpreter between unloading and reloading modules.
    for number in forwarded_signals()[1:]:
        signal.signal(number, interrupted)
    try:
        sys.exit(main())
    except (Refusal, OSError, ValueError, KeyError) as error:
        print(f'kernel: {error}', file=sys.stderr)
        sys.exit(2)
PY
