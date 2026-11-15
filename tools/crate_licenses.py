#!/usr/bin/env python3
"""Bundle the locked evaluation crates' license files for redistribution."""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

LICENSE_FILE = re.compile(r"^(licen[cs]e|copying|notice|unlicense|copyright)([-_.].*)?$", re.IGNORECASE)


def shipped_packages(metadata, root_name):
    packages = {package["id"]: package for package in metadata["packages"]}
    nodes = {node["id"]: node for node in metadata["resolve"]["nodes"]}
    roots = [id for id, package in packages.items() if package["name"] == root_name and package["source"] is None]
    if len(roots) != 1:
        raise SystemExit(f"crate_licenses.py: expected one local package {root_name}, found {len(roots)}")
    seen, stack = set(), roots[:]
    while stack:
        id = stack.pop()
        if id in seen:
            continue
        seen.add(id)
        for dep in nodes[id]["deps"]:
            # kind None: a normal dependency, compiled into the binary (proc macros included: their output is).
            if any(kind["kind"] is None for kind in dep["dep_kinds"]):
                stack.append(dep["pkg"])
    seen.difference_update(roots)
    return sorted((packages[id] for id in seen), key=lambda package: (package["name"], package["version"]))


def license_files(package):
    directory = Path(package["manifest_path"]).parent
    files = {path for path in directory.iterdir() if path.is_file() and LICENSE_FILE.match(path.name)}
    if package.get("license_file"):
        named = directory / package["license_file"]
        if named.is_file():
            files.add(named)
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cargo", default="cargo")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--target", required=True, help="the target triple cmt-evaluate is built for")
    parser.add_argument("--package", default="cmt-evaluate")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    metadata = json.loads(subprocess.run(
        [args.cargo, "metadata", "--format-version", "1", "--locked", "--offline", "--filter-platform", args.target,
         "--manifest-path", str(args.manifest)], check=True, capture_output=True, text=True).stdout)
    packages = shipped_packages(metadata, args.package)

    missing = [f"{package['name']} {package['version']}" for package in packages if not license_files(package)]
    if missing:
        raise SystemExit("crate_licenses.py: no license file in " + ", ".join(missing))
    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    index = []
    for package in packages:
        destination = args.out / f"{package['name']}-{package['version']}"
        destination.mkdir()
        for path in license_files(package):
            shutil.copyfile(path, destination / path.name)
        index.append(f"{package['name']} {package['version']}: {package.get('license') or 'see license file'}")
    (args.out / "index.txt").write_text(
        "Rust crates compiled into cmt-evaluate (evaluation/Cargo.lock), with the license each declares.\n"
        "Each directory holds that crate's license files as published.\n\n" + "\n".join(index) + "\n")
    print(f"crate_licenses.py: {len(packages)} crates -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
