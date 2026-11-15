#!/usr/bin/env python3
"""Convert a manifest of local tunes and atomically replace the site's database."""

import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="TSV with file, id, machine, tier, status, note columns")
    parser.add_argument("--bin", type=Path, default=Path("target/release/cudamemcpy-site"))
    parser.add_argument("--db", type=Path, default=Path("data/cmx.duckdb"))
    args = parser.parse_args()
    binary, database = args.bin.resolve(), args.db.resolve()
    if not binary.is_file():
        parser.error(f"build the site first; missing {binary}")
    with args.manifest.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["file", "id", "machine", "tier", "status", "note"]:
            parser.error("manifest columns must be file, id, machine, tier, status, note")
        rows = list(reader)
    if not rows:
        parser.error("manifest has no runs; existing database left intact")
    if any(None in row or None in row.values() or not row["file"] or not row["id"] for row in rows):
        parser.error("each row needs six columns, with a nonempty file and id")
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        parser.error("manifest contains duplicate run IDs")
    database.parent.mkdir(parents=True, exist_ok=True)
    # Same filesystem as the destination, so os.replace is atomic. Any conversion
    # or import failure exits before the existing database is replaced.
    with tempfile.TemporaryDirectory(prefix=".import-", dir=database.parent) as scratch:
        work = Path(scratch)
        converted = []
        for index, row in enumerate(rows):
            source = args.manifest.resolve().parent / row["file"]
            output = work / f"{index}.json"
            command = [str(binary), "convert", str(source), "--id", row["id"], "-o", str(output)]
            for key in ("machine", "tier", "status", "note"):
                if row.get(key):
                    command += [f"--{key}", row[key]]
            subprocess.run(command, check=True)
            converted.append(str(output))
        candidate = work / "runs.duckdb"
        env = dict(os.environ, CMX_DB=str(candidate))
        subprocess.run([str(binary), "import", *converted], env=env, check=True)
        os.replace(candidate, database)
    print(f"Imported {len(rows)} runs into {database}; run scripts/build.sh to publish a new snapshot.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"import: {error}", file=sys.stderr)
        sys.exit(1)
