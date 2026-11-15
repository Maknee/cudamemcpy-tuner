#!/usr/bin/env python3
"""libcuda release checks for CI (.github/workflows/libcuda.yml). Standard library only.

  latest  Find every NVIDIA Linux driver release and check that the hook has a row for each one that matters.
  load    Preload the built hook into a process that loads each pinned libcuda, and check that it patches.

latest: where releases come from
  apt   NVIDIA's CUDA apt repositories ubuntu2404/x86_64 and ubuntu2404/sbsa, package libnvidia-compute(-<branch>).
        InRelease must carry a valid signature by NVIDIA's repository key (NVIDIA_KEY, the key drivers.manifest
        documents; checked with gpgv), Packages.gz must match InRelease's SHA256, and each .deb the SHA256 in
        Packages.gz. ubuntu2204 lists the same versions (2026-11-12).
  run   download.nvidia.com/XFree86/Linux-<arch>/: one directory per .run installer release. Some releases are only
        here (drivers.manifest names them). The listing is unsigned, so it only names versions: a .run is used only
        when NVIDIA's <url>.sha256sum exists, and only with that hash (the trust sigscan's fetch applies).
  tags  github.com/NVIDIA/open-gpu-kernel-modules tags: every release with open kernel modules, including ones with
        no public user-space download. Informational: a tag in neither source above is listed, never fails.
  A new release is normally in apt, the listing, or both within days; the union of the two is what is checked.

latest: which releases are checked (POLICY)
  x86_64   every release from the oldest table/scanned row upward: drivers.manifest covers them all.
  aarch64  the newest release of every branch that has a table/scanned row, and of every newer branch: the
           aarch64 table covers selected releases.
  A `survey` row acknowledges a release without a hook row: reported, not a failure.

latest: states
  table    a hardware-checked row exists; the pinned job (`cmt-sigscan verify`, `load`) checks it
  scanned  a static-scan-only row exists (the hook uses it only with MEMCPY_HOOK_ALLOW_UNVERIFIED_ROWS=1)
  survey   acknowledged in drivers.manifest without a hook row
  missing  no row; cmt-sigscan finds every patch site. The report prints the drivers.manifest line to add
  broken   no row; cmt-sigscan cannot find the patch sites, or the package does not unpack as expected
  respun   NVIDIA republished a pinned version with a different libcuda or installer (same version string)
  unavailable  listed, but no downloadable package with a published hash: a warning, retried next run
  missing, broken and respun fail (exit 1). Without a row the hook still interposes the driver API, but its in-driver
  patches stay off for that release.

  For a new release the package is downloaded, libcuda is extracted here (dpkg-deb / the .run's makeself payload,
  never executed), then extracted again by `cmt-sigscan fetch` from the printed manifest line, which must give the
  same SHA-256: the line is proven to work before anyone copies it.

load: what it shows
  For each pinned table/scanned row of this machine's architecture, a child process (/bin/true) starts with
  LD_PRELOAD=<hook> and LD_LIBRARY_PATH pointing at a libcuda.so.1 symlink to the pinned libcuda.so.<version>. The
  hook's constructor dlopens it (no cuInit, no GPU, no kernel module), selects the row by the file name, resolves the
  driver API and installs the in-driver patches: InstallPatches compares the row's original bytes at each site with
  the loaded image before it writes. The check requires the row to be selected, every patch site the row names to be
  patched, and no error or missing driver symbol. It does not show that the patched code copies correctly: that
  needs a GPU (the gpu-labelled tests).

Exit codes: 0 pass, 1 a release or row fails, 2 the check itself could not run (network, signature, tool).
"""
import argparse
import gzip
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

APT = {  # architecture -> NVIDIA's CUDA apt repository for it
    "x86_64": "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64",
    "aarch64": "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa",
}
APT_KEY = "3bf863cc.pub"  # served next to the index; trusted only through NVIDIA_KEY below
NVIDIA_KEY = "EB693B3035CD5710E231E123A4B469963BF863CC"  # cuda-archive-keyring, key 3bf863cc
RUN_MIRRORS = ("https://download.nvidia.com/XFree86", "https://us.download.nvidia.com/XFree86")
TAGS = "https://api.github.com/repos/NVIDIA/open-gpu-kernel-modules/tags?per_page=100&page={}"
POLICY = {"x86_64": "every", "aarch64": "newest-per-branch"}
DEB_ARCH = {"x86_64": "amd64", "aarch64": "arm64"}
FAILING = ("missing", "broken", "respun")
MAX_INDEX = 64 << 20
MAX_PACKAGE = 2 << 30  # cmt-sigscan's limits
MAX_LIBRARY = 1 << 30
USER_AGENT = "cmt-libcuda-check"


class CheckError(Exception):
    """The check could not run (network, signature, tool): exit 2, not a verdict about a driver."""


class Unavailable(Exception):
    """A listed release has no downloadable package (HTTP 404): a warning, not a verdict."""


def version_key(version):
    return tuple(int(x) for x in version.split("."))


def branch(version):
    return int(version.split(".")[0])


def is_version(text):
    return re.fullmatch(r"\d+\.\d+(\.\d+)?", text) is not None


def warn(message):
    # A workflow command: GitHub shows it as an annotation on the run.
    print(f"::warning::{message}", flush=True)


# Network

def request(url, method="GET"):
    if not url.startswith("https://"):
        raise CheckError(f"refusing non-HTTPS URL {url}")
    return urllib.request.Request(url, method=method, headers={"User-Agent": USER_AGENT, **auth(url)})


def auth(url):
    token = os.environ.get("GITHUB_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token and url.startswith("https://api.github.com/") else {}


class SafeRedirects(urllib.request.HTTPRedirectHandler):
    """Redirects stay on HTTPS, and the GitHub token never follows one to another host (urllib's default handler
    follows a redirect to http:// and copies every header, Authorization included, to the new URL)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme != "https":
            raise CheckError(f"{req.full_url}: refusing redirect to non-HTTPS {newurl}")
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and origin(newurl) != origin(req.full_url):
            new.remove_header("Authorization")
        return new


def origin(url):
    parts = urllib.parse.urlsplit(url)
    return parts.scheme, parts.hostname, parts.port


OPENER = urllib.request.build_opener(SafeRedirects)


def retried(action, url):
    """action() with three attempts for transient errors; HTTP 404 raises Unavailable at once."""
    for attempt in range(3):
        try:
            return action()
        except urllib.error.HTTPError as error:
            if error.code == 404:
                raise Unavailable(f"{url}: HTTP 404") from error
            last = error
        except (OSError, ValueError) as error:  # URLError, timeouts, truncated reads
            last = error
        time.sleep(5 * (attempt + 1))
    raise CheckError(f"{url}: {last}")


def fetch(url, limit=MAX_INDEX):
    def once():
        with OPENER.open(request(url), timeout=120) as response:
            data = response.read(limit + 1)
        if len(data) > limit:
            raise CheckError(f"{url}: larger than {limit} bytes")
        return data
    return retried(once, url)


def exists(url):
    def once():
        with OPENER.open(request(url, "HEAD"), timeout=60):
            return True
    try:
        return retried(once, url)
    except Unavailable:
        return False


def download(url, path, sha256):
    """Stream url to path within MAX_PACKAGE; the file is kept only if its SHA-256 is sha256."""
    def once():
        digest, size = hashlib.sha256(), 0
        with OPENER.open(request(url), timeout=600) as response, open(path, "wb") as out:
            while chunk := response.read(1 << 20):
                size += len(chunk)
                if size > MAX_PACKAGE:
                    raise CheckError(f"{url}: larger than {MAX_PACKAGE} bytes")
                digest.update(chunk)
                out.write(chunk)
        return digest.hexdigest()
    actual = retried(once, url)
    if actual != sha256:
        path.unlink()
        raise CheckError(f"{url}: SHA-256 {actual} differs from the published {sha256}")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


# drivers.manifest and the generated tables

def read_manifest(path):
    """[{arch, version, url, sha256, member, library, purpose}] in file order."""
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        fields = line.split("#")[0].split()
        if not fields:
            continue
        if len(fields) != 7 or fields[0] not in APT or not is_version(fields[1]):
            raise CheckError(f"{path}:{number}: malformed row")
        rows.append(dict(zip(("arch", "version", "url", "sha256", "member", "library", "purpose"), fields)))
    return rows


TABLE_FILES = {"x86_64": "src/gen/versions_db.inc", "aarch64": "src/gen/versions_aarch64.inc"}


def table_sites(repo, arch):
    """{version: {site: offset}} from the generated table: which patches a row asks the hook to install."""
    text = (repo / TABLE_FILES[arch]).read_text()
    rows = {}
    for block in re.split(r"\n \{\n", text)[1:]:
        version = re.search(r'\.version = "([0-9.]+)"', block)
        sites = dict(re.findall(r"\.(h2h_va|buffer_init_va|internal_memcpy_va) = (0x[0-9a-f]+),", block))
        if not version or len(sites) != 3:
            raise CheckError(f"{TABLE_FILES[arch]}: cannot read a row")
        rows[version.group(1)] = {k: int(v, 16) for k, v in sites.items()}
    if not rows:
        raise CheckError(f"{TABLE_FILES[arch]}: no rows")
    return rows


# Sources

def gpg_keyring(work):
    """NVIDIA's apt key, dearmored for gpgv. Downloaded over HTTPS; trusted only through NVIDIA_KEY."""
    keyring = work / "nvidia-apt.gpg"
    if not keyring.exists():
        armored = fetch(f"{APT['x86_64']}/{APT_KEY}", 1 << 20)
        run = subprocess.run(["gpg", "--dearmor"], input=armored, capture_output=True)
        if run.returncode != 0:
            raise CheckError(f"gpg --dearmor: {run.stderr.decode(errors='replace')}")
        keyring.write_bytes(run.stdout)
    return keyring


def signed_release(repo, keyring, work):
    """InRelease's signed text, after gpgv reports a valid signature whose primary key is NVIDIA_KEY."""
    path = work / (re.sub(r"\W", "_", repo) + ".InRelease")
    path.write_bytes(fetch(f"{repo}/InRelease"))
    # Status lines go to stderr (fd 2); the verified payload alone goes to stdout.
    run = subprocess.run(["gpgv", "--status-fd", "2", "--keyring", str(keyring), "--output", "-", str(path)],
                         capture_output=True)
    status = run.stderr.decode(errors="replace")
    # VALIDSIG <fingerprint> ... <primary key fingerprint>: a subkey signature names the primary last.
    primaries = {line.split()[-1] for line in status.splitlines() if line.startswith("[GNUPG:] VALIDSIG ")}
    if run.returncode != 0 or primaries != {NVIDIA_KEY}:
        raise CheckError(f"{repo}/InRelease: no valid signature by {NVIDIA_KEY}\n{status}")
    return run.stdout.decode()


def apt_releases(arch, keyring, work):
    """{version: [stanza, ...]} of libnvidia-compute(-<branch>) packages in the signed apt index."""
    repo = APT[arch]
    release = signed_release(repo, keyring, work)
    date = re.search(r"^Date: (.+)$", release, re.M)
    if date:
        age = time.time() - time.mktime(time.strptime(date.group(1).strip(), "%a, %d %b %Y %H:%M:%S %z"))
        if age > 45 * 86400:
            warn(f"{repo}/InRelease is {int(age // 86400)} days old: the index may be stale")
    want, in_sha256 = None, False
    for line in release.splitlines():
        if not line.startswith(" "):
            in_sha256 = line.startswith("SHA256:")
        elif in_sha256 and len(line.split()) == 3 and line.split()[2] == "Packages.gz":
            want = line.split()[0]
    if want is None:
        raise CheckError(f"{repo}/InRelease lists no SHA256 for Packages.gz")
    data = fetch(f"{repo}/Packages.gz")
    if hashlib.sha256(data).hexdigest() != want:
        raise CheckError(f"{repo}/Packages.gz differs from the hash in the signed InRelease")
    found = {}
    for block in gzip.decompress(data).decode().split("\n\n"):
        stanza = dict(re.findall(r"^([A-Za-z0-9-]+): *(.*)$", block, re.M))
        match = re.fullmatch(r"(?:\d+:)?(\d+\.\d+(?:\.\d+)?)-[0-9A-Za-z.+~]+", stanza.get("Version", ""))
        if not re.fullmatch(r"libnvidia-compute(-\d+)?", stanza.get("Package", "")) or not match:
            continue
        if stanza.get("Architecture") != DEB_ARCH[arch] or not re.fullmatch(r"[0-9a-f]{64}", stanza.get("SHA256", "")):
            continue
        filename = stanza.get("Filename", "").removeprefix("./")
        if not re.fullmatch(r"[A-Za-z0-9._+~-]+\.deb", filename):
            continue
        stanza["url"] = f"{repo}/{filename}"
        found.setdefault(match.group(1), []).append(stanza)
    return found


def run_releases(arch):
    """Versions with a directory in download.nvidia.com/XFree86/Linux-<arch>/ (unsigned: names only)."""
    text = fetch(f"{RUN_MIRRORS[0]}/Linux-{arch}/").decode(errors="replace")
    return {v for v in re.findall(r"href=['\"]?(\d+\.\d+(?:\.\d+)?)/['\"]?", text)}


def run_package(arch, version):
    """(url, sha256) of the .run installer with NVIDIA's published hash; Unavailable if no mirror has both."""
    path = f"Linux-{arch}/{version}/NVIDIA-Linux-{arch}-{version}.run"
    for mirror in RUN_MIRRORS:
        url = f"{mirror}/{path}"
        try:
            published = fetch(f"{url}.sha256sum", 4096).decode(errors="replace").split()
        except Unavailable:
            continue
        if published and re.fullmatch(r"[0-9a-f]{64}", published[0]) and exists(url):
            return url, published[0]
    raise Unavailable(f"no .run with a .sha256sum for {arch} {version}")


def tag_releases():
    """Version tags of NVIDIA/open-gpu-kernel-modules (informational)."""
    tags = set()
    for page in range(1, 4):
        names = [t.get("name", "") for t in json.loads(fetch(TAGS.format(page)))]
        tags |= {n for n in names if is_version(n)}
        if len(names) < 100:
            break
    return tags


# Packages -> libcuda

def tar_member(stream, member, out):
    """Copy the one regular file `member` of the tar stream to out; nothing is unpacked to disk."""
    wanted = os.path.normpath(member.removeprefix("./"))
    found = False
    with tarfile.open(fileobj=stream, mode="r|*") as archive:
        for entry in archive:
            if os.path.normpath(entry.name.removeprefix("./")) != wanted:
                continue
            if found or not entry.isreg() or entry.size > MAX_LIBRARY:
                raise ValueError(f"{member}: duplicate, not a regular file, or larger than {MAX_LIBRARY}")
            shutil.copyfileobj(archive.extractfile(entry), out)
            found = True
    if not found:
        raise ValueError(f"archive member not found: {member}")


def run_payload(path, arch, version):
    """(offset, decompressor) of a .run's makeself payload, read from its header as text (never executed)."""
    with open(path, "rb") as f:
        header = f.read(65536).decode("latin-1")

    def field(pattern):
        values = re.findall(pattern, header, re.M)
        if len(values) != 1:
            raise ValueError(f"installer header needs one field matching {pattern}")
        return values[0]
    skip = int(field(r"^skip=([0-9]+)$"))
    decompress_lines = int(field(r"^skip_decompress=([0-9]+)$")) + int(field(r"^size_decompress=([0-9]+)$"))
    if field(r"^version_string=([0-9.]+)$") != version or field(r'^TARGET_ARCH="([a-z0-9_]+)"$') != arch:
        raise ValueError("installer version/architecture differs from the release")
    if not 0 < skip <= 1_000_000 or decompress_lines != skip:
        raise ValueError("unsupported installer payload boundary")
    compression = field(r'^ *echo "  Compression +: ([a-z0-9]+)"$')
    magic = {"xz": b"\xfd7zXZ\0", "zstd": b"\x28\xb5\x2f\xfd"}.get(compression)
    if magic is None:
        raise ValueError(f"unsupported installer compression {compression}")
    with open(path, "rb") as f:
        for _ in range(skip - 1):
            if not f.readline():
                raise ValueError("installer ends before payload")
        offset = f.tell()
        if f.read(len(magic)) != magic:
            raise ValueError("installer payload has the wrong compression magic")
    return offset, ["xz", "-dc"] if compression == "xz" else ["zstd", "-dc"]


def extract(package, arch, version, member, library):
    """Extract `member` (libcuda) from a .deb or .run into `library`; ValueError if the package is not as expected."""
    if package.suffix == ".deb":
        command, stdin = ["dpkg-deb", "--fsys-tarfile", str(package)], subprocess.DEVNULL
    else:
        offset, command = run_payload(package, arch, version)
        stdin = open(package, "rb")
        stdin.seek(offset)
    try:
        with subprocess.Popen(command, stdin=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as child, \
                open(library, "wb") as out:
            try:
                tar_member(child.stdout, member, out)
            finally:
                child.stdout.read()  # let the decompressor finish so its exit status means something
                error = child.stderr.read().decode(errors="replace")
        if child.returncode != 0:
            raise ValueError(f"{command[0]} failed: {error.strip()}")
    finally:
        if stdin is not subprocess.DEVNULL:
            stdin.close()


def candidate_library(arch, version, url, sha256, member, work):
    """Download one package, extract its libcuda, return (package path, library path, library sha256)."""
    directory = work / "packages" / arch / version
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True)
    package = directory / url.rsplit("/", 1)[1]
    print(f"  downloading {url}", flush=True)
    download(url, package, sha256)
    library = directory / f"libcuda.so.{version}"
    extract(package, arch, version, member, library)
    return package, library, sha256_file(library)


def deb_member(arch, version):
    return f"./usr/lib/{arch}-linux-gnu/libcuda.so.{version}"


# latest

def checked_releases(arch, releases, rows):
    """The releases POLICY says must be accounted for, newest first."""
    hooked = [r["version"] for r in rows if r["arch"] == arch and r["purpose"] in ("table", "scanned")]
    if not hooked:
        raise CheckError(f"drivers.manifest has no {arch} table or scanned row")
    if POLICY[arch] == "every":
        floor = min(map(version_key, hooked))
        chosen = {v for v in releases if version_key(v) >= floor}
    else:
        branches = {branch(v) for v in hooked}
        newest = {}
        for v in releases:
            b = branch(v)
            if (b in branches or b > max(branches)) and (b not in newest or version_key(v) > version_key(newest[b])):
                newest[b] = v
        chosen = set(newest.values())
    return sorted(chosen, key=version_key, reverse=True)


def scan_new(arch, version, apt, args, repo):
    """A release without a manifest row: (state, manifest line or None, detail)."""
    if apt:
        stanza = max(apt, key=lambda s: s["Version"])  # the newest Debian revision
        url, sha256, member = stanza["url"], stanza["SHA256"], deb_member(arch, version)
    else:
        url, sha256 = run_package(arch, version)
        member = f"./libcuda.so.{version}"
    try:
        package, library, library_sha = candidate_library(arch, version, url, sha256, member, args.work)
    except ValueError as error:
        return "broken", None, f"{url}: {error}"
    line = f"{arch}  {version}  {url}  {sha256}  {member}  {library_sha}  scanned"
    # The line must work for cmt-sigscan itself: its own extractor, from the downloaded package, must agree.
    cache = args.work / "sigscan-drivers"
    manifest = args.work / "candidate.manifest"
    manifest.write_text(line + "\n")
    pkg = cache / arch / version / "pkg"
    shutil.rmtree(cache / arch / version, ignore_errors=True)
    pkg.mkdir(parents=True)
    shutil.move(package, pkg / package.name)
    fetched = subprocess.run([str(args.sigscan), "--repo", str(repo), "--manifest", str(manifest), "--drivers",
                              str(cache), "fetch", "--arch", arch, "--versions", version],
                             capture_output=True, text=True)
    if fetched.returncode != 0:
        return "broken", line, f"cmt-sigscan fetch of the proposed line failed: {fetched.stderr.strip()}"
    scan = subprocess.run([str(args.sigscan), "scan", "--arch", arch, "--format", "rows", str(library)],
                          capture_output=True, text=True)
    shutil.rmtree(cache / arch / version, ignore_errors=True)
    shutil.rmtree(library.parent, ignore_errors=True)  # 100+ MB each
    if scan.returncode != 0:
        return "broken", line, scan.stderr.strip() or f"cmt-sigscan scan exited {scan.returncode}"
    return "missing", line, "cmt-sigscan scan finds every patch site"


def check_pins(arch, rows, apt, work):
    """Pinned rows whose version NVIDIA republished: [(version, state, detail)]."""
    results = []
    for row in rows:
        if row["arch"] != arch or row["purpose"] == "survey":
            continue
        version, url = row["version"], row["url"]
        others = [s for s in apt.get(version, []) if s["url"] != url]
        if url.endswith(".deb"):
            pinned = [s for s in apt.get(version, []) if s["url"] == url]
            if not pinned:
                warn(f"{arch} {version}: {url} is no longer in the apt index (cached copies still verify)")
            elif pinned[0]["SHA256"] != row["sha256"]:
                results.append((version, "respun", f"{url}: index SHA256 {pinned[0]['SHA256']} != pinned"))
        else:
            try:
                published = fetch(f"{url}.sha256sum", 4096).decode(errors="replace").split()
            except Unavailable:
                warn(f"{arch} {version}: {url}.sha256sum is gone (cached copies still verify)")
                published = [row["sha256"]]
            if published[:1] != [row["sha256"]]:
                results.append((version, "respun", f"{url}.sha256sum now says {' '.join(published[:1]) or 'nothing'}"))
        # Another package of the same version (a new Debian revision, or apt for a .run-only release): its libcuda
        # must be the pinned one, since the hook selects rows by version string alone.
        for stanza in others:
            print(f"{arch} {version}: comparing {stanza['url']} with the pinned libcuda", flush=True)
            try:
                _, library, library_sha = candidate_library(arch, version, stanza["url"], stanza["SHA256"],
                                                            deb_member(arch, version), work)
                shutil.rmtree(library.parent, ignore_errors=True)
            except ValueError as error:
                results.append((version, "respun", f"{stanza['url']}: {error}"))
                continue
            if library_sha != row["library"]:
                results.append((version, "respun", f"{stanza['url']}: libcuda {library_sha} != pinned {row['library']}"))
    return results


def latest(args):
    repo = args.repo.resolve()
    args.work.mkdir(parents=True, exist_ok=True)
    args.work = args.work.resolve()
    rows = read_manifest(repo / "tools/sigscan/drivers.manifest")
    keyring = args.keyring or gpg_keyring(args.work)
    try:
        tags = tag_releases()
    except (CheckError, Unavailable, ValueError) as error:
        warn(f"open-gpu-kernel-modules tags unavailable: {error}")
        tags = set()
    report, failed, everything = [], False, set()
    for arch in args.arch or sorted(APT):
        apt = apt_releases(arch, keyring, args.work)
        runs = run_releases(arch)
        known = {r["version"]: r["purpose"] for r in rows if r["arch"] == arch}
        floor = min(version_key(v) for v in known)
        # A parser or source that silently returns nothing must not read as "no new releases".
        if not any(version_key(v) >= floor for v in apt):
            raise CheckError(f"{APT[arch]}: no libnvidia-compute package at or above {min(known, key=version_key)}")
        if not any(version_key(v) >= floor for v in runs):
            raise CheckError(f"{RUN_MIRRORS[0]}/Linux-{arch}/: no release directory at or above the manifest")
        for row in rows:
            if row["arch"] == arch and "/XFree86/" in row["url"] and row["version"] not in runs:
                warn(f"{arch} {row['version']}: pinned .run release no longer in the Linux-{arch} listing")
        releases = set(apt) | runs
        everything |= releases
        checked = checked_releases(arch, releases, rows)
        newest = {}
        for v in checked:
            newest.setdefault(branch(v), v)
        for version, state, detail in check_pins(arch, rows, apt, args.work):
            report.append((arch, version, state, None, detail))
            failed = True
            print(f"{arch} {version}: {state}: {detail}", flush=True)
        for version in checked:
            purpose = known.get(version)
            line, detail = None, ""
            if purpose:
                state = purpose
            else:
                print(f"{arch} {version}: no drivers.manifest row; scanning", flush=True)
                try:
                    state, line, detail = scan_new(arch, version, apt.get(version), args, repo)
                except Unavailable as error:
                    state, detail = "unavailable", str(error)
                    warn(f"{arch} {version}: {detail}")
            failed |= state in FAILING
            if state not in ("table", "scanned", "survey") or newest.get(branch(version)) == version:
                report.append((arch, version, state, line, detail))
            print(f"{arch} {version}: {state}" + (f": {detail}" if detail else ""), flush=True)
        print(f"{arch}: {len(releases)} releases listed ({len(apt)} apt, {len(runs)} .run), "
              f"{len(checked)} checked ({POLICY[arch]})", flush=True)
    floor = min(version_key(r["version"]) for r in rows)
    tag_only = sorted((t for t in tags - everything if version_key(t) >= floor), key=version_key, reverse=True)
    summary = render(report, tag_only, failed)
    print("\n" + summary)
    if args.summary:
        with open(args.summary, "a") as out:
            out.write(summary)
    return 1 if failed else 0


def render(report, tag_only, failed):
    md = ["## libcuda releases: " + ("action needed" if failed else "every checked release has a row"), "",
          "Newest release of each branch, and every release that needs attention.", "",
          "| arch | version | state |", "|---|---|---|"]
    md += [f"| {a} | {v} | {s} |" for a, v, s, _, _ in report]
    for arch, version, state, line, detail in report:
        if state == "missing":
            md += ["", f"### {arch} {version}: missing", "",
                   "cmt-sigscan finds every patch site. Add this line to tools/sigscan/drivers.manifest (its extraction "
                   "was checked with `cmt-sigscan fetch`), run `cmt-sigscan fetch` and `cmt-sigscan generate`, review "
                   "the table diff:", "", "```", line, "```"]
        elif state in ("broken", "respun", "unavailable"):
            md += ["", f"### {arch} {version}: {state}", "", "```", detail, "```"]
            if line:
                md += ["", "The package it was checked from (not a row to add while the scanner rejects it):", "",
                       "```", line, "```"]
    if tag_only:
        md += ["", "open-gpu-kernel-modules tags with no apt package or .run release (no public libcuda to check): "
               + ", ".join(tag_only)]
    return "\n".join(md) + "\n"


# load

KNOB_LINE = re.compile(r"\[cmt\] (knob |[A-Z][A-Z0-9_]+ {2,})")


def load(args):
    repo = args.repo.resolve()
    arch = platform.machine()
    if arch not in APT:
        raise CheckError(f"unsupported machine {arch}")
    hook = args.hook.resolve()
    sites = table_sites(repo, arch)
    rows = [r for r in read_manifest(repo / "tools/sigscan/drivers.manifest")
            if r["arch"] == arch and r["purpose"] in ("table", "scanned")]
    if set(sites) != {r["version"] for r in rows}:
        raise CheckError(f"{TABLE_FILES[arch]} and drivers.manifest disagree: run cmt-sigscan verify")
    if args.versions:
        rows = [r for r in rows if r["version"] in args.versions]
        if {r["version"] for r in rows} != set(args.versions):
            raise CheckError(f"--versions names a version with no {arch} table/scanned row")
    expect = {"h2h_va": "patched h2h_engine at ", "buffer_init_va": "wrapped staging_buffer_init at ",
              "internal_memcpy_va": "patched internal_memcpy_temporal at "}
    failures = 0
    with tempfile.TemporaryDirectory(prefix="cmt-load-") as scratch:
        for row in rows:
            version = row["version"]
            library = (args.drivers / arch / version / f"libcuda.so.{version}").resolve()
            if not library.is_file():
                raise CheckError(f"{library}: not fetched (cmt-sigscan fetch)")
            if sha256_file(library) != row["library"]:
                raise CheckError(f"{library}: SHA-256 differs from drivers.manifest")
            # libcuda.so.1 -> the pinned file: the hook reads the version from the name realpath() gives.
            directory = Path(scratch) / version
            directory.mkdir()
            (directory / "libcuda.so.1").symlink_to(library)
            # Request every patch the row has a site for: a global copy loop asks for the descriptor and byte copy
            # replacements, a directional one for the descriptor copy alone (aarch64 rows have no byte copy loop;
            # requesting it would log an error by design); a staging size asks for the staging wrapper. Scanned
            # and aarch64 rows need the opt-in.
            loop = "STAGING_COPY_NT_IMPL" if sites[version]["internal_memcpy_va"] else "STAGING_COPY_NT_IMPL_H2D"
            env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LD_LIBRARY_PATH": str(directory),
                   "LD_PRELOAD": str(hook), "MEMCPY_HOOK_DEBUG": "1", loop: "memcpy", "STAGING_BUFFER_SIZE": "4096",
                   "MEMCPY_HOOK_ALLOW_UNVERIFIED_ROWS": "1"}
            run = subprocess.run(["/bin/true"], env=env, capture_output=True, text=True, timeout=300)
            log = run.stderr
            problems = []
            if run.returncode != 0:
                problems.append(f"exit status {run.returncode}")
            selected = re.search(rf"\[cmt\] libcuda (\S+) at base .*\(version '{re.escape(version)}', {arch} db row: "
                                 rf"{re.escape(version)}[;,]", log)
            if not selected or not selected.group(1).startswith(str(directory) + "/"):
                problems.append(f"the hook did not load {directory}/libcuda.so.1 and select row {version}")
            for site, text in expect.items():
                wanted = sites[version][site] != 0
                if (text in log) != wanted:
                    problems.append(f"{text.split(' at ')[0]}: expected {'a' if wanted else 'no'} patch")
            # The configuration dump (one "[cmt] NAME  value ..." line per knob) is not part of the verdict.
            events = [x for x in log.splitlines() if not KNOB_LINE.match(x)]
            problems += [x.strip() for x in events if re.search(r"error:|missing libcuda symbol|[Cc]annot ", x)]
            if problems:
                failures += 1
                print(f"{arch} {version}: FAIL\n  " + "\n  ".join(problems) + "\n--- hook log ---\n" +
                      "\n".join(events), flush=True)
            else:
                patched = [t.split(" at ")[0].split()[-1] for t in expect.values() if t in log]
                print(f"{arch} {version}: ok ({row['purpose']} row; {', '.join(patched)})", flush=True)
    print(f"{len(rows) - failures}/{len(rows)} {arch} rows load and patch")
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", type=Path, default=Path("."), help="repository root (default: .)")
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("latest", help="find NVIDIA releases and check each has a row")
    one.add_argument("--sigscan", type=Path, required=True, help="built cmt-sigscan")
    one.add_argument("--work", type=Path, required=True, help="download directory (needs ~1 GB per new release)")
    one.add_argument("--keyring", type=Path, help="gpgv keyring with NVIDIA's apt key (default: download it)")
    one.add_argument("--arch", choices=sorted(APT), action="append", help="default: both")
    one.add_argument("--summary", type=Path, help="append the Markdown report here (GITHUB_STEP_SUMMARY)")
    two = sub.add_parser("load", help="preload the hook into processes that load each pinned libcuda")
    two.add_argument("--hook", type=Path, required=True, help="built libcmt_hook.so")
    two.add_argument("--drivers", type=Path, required=True, help="cmt-sigscan's driver cache")
    two.add_argument("--versions", nargs="+", help="only these versions (default: every table/scanned row)")
    args = parser.parse_args()
    return latest(args) if args.command == "latest" else load(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (CheckError, Unavailable, subprocess.CalledProcessError, OSError) as error:
        print(f"libcuda_check: {error}", file=sys.stderr)
        sys.exit(2)
