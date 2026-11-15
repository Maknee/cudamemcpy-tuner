#!/usr/bin/env bash
# Package a configured and built tree as the release tarball, and check what goes in it.
# Usage: package.sh BUILD_DIR NAME OUT_DIR
#   Writes OUT_DIR/NAME.tar.gz (one top-level directory, cmt/) and OUT_DIR/NAME.tar.gz.sha256.
# Environment:
#   SOURCE_DATE_EPOCH   file times in the tarball (default: the commit time of HEAD)
#   GLIBC_FLOOR, GLIBCXX_FLOOR, CXXABI_FLOOR   highest symbol versions a shipped ELF file may require
#
# Contents: the `cmt` install component only (README.md, "Build from source"): cmt, gpu_copy, cpu_loops,
# libcmt_hook.so, libcmt_trace.so, kernel_patches/, LICENSE, THIRD_PARTY_NOTICES, and licenses/ (the license files
# of the libraries compiled in). The tarball is reproducible: names sorted, owner 0, every mtime SOURCE_DATE_EPOCH,
# no gzip timestamp.
set -euo pipefail

build=${1:?usage: package.sh BUILD_DIR NAME OUT_DIR}
name=${2:?usage: package.sh BUILD_DIR NAME OUT_DIR}
out=${3:?usage: package.sh BUILD_DIR NAME OUT_DIR}
here=$(cd "$(dirname "$0")" && pwd)
: "${SOURCE_DATE_EPOCH:=$(git log -1 --format=%ct)}"

mkdir -p "$out"
out=$(cd "$out" && pwd)
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
cmake --install "$build" --prefix "$stage/cmt" --component cmt >/dev/null

fail() {
	echo "::error::package: $*"
	exit 1
}

# Exactly the documented top-level entries.
expected="LICENSE THIRD_PARTY_NOTICES cmt cpu_loops gpu_copy kernel_patches libcmt_hook.so libcmt_trace.so licenses"
actual=$(find "$stage/cmt" -mindepth 1 -maxdepth 1 -printf '%f\n' | LC_ALL=C sort | xargs)
[[ $actual == "$expected" ]] || fail "top level is '$actual', expected '$expected'"
for file in kernel_patches/nvidia-open/kmod.sh kernel_patches/nvidia-open/releases.json kernel_patches/nvidia-open/COPYING; do
	[[ -s $stage/cmt/$file ]] || fail "no $file"
done
[[ -n $(find "$stage/cmt/kernel_patches/nvidia-open" -name '*.patch' -print -quit) ]] || fail "no kernel patches"
# Every fetched library compiled into a shipped binary, with at least one non-empty license file each.
for library in abseil benchmark capstone cli11 expected fmt function2 indicators picosha2 simdjson spdlog unordered_dense; do
	[[ -n $(find "$stage/cmt/licenses/$library" -type f -size +0 -print -quit 2>/dev/null) ]] ||
		fail "no license files for $library"
done
[[ ! -e $stage/cmt/licenses/keystone ]] || fail "Keystone is not shipped; its license does not belong in licenses/"

# Strip ELF files, then hold them to the stated floors.
elf=()
while IFS= read -r -d '' file; do
	if [[ $(head -c 4 "$file" | od -An -c | tr -d ' ') == '177ELF' ]]; then
		strip --strip-unneeded "$file"
		elf+=("$file")
	fi
done < <(find "$stage/cmt" -type f -print0 | LC_ALL=C sort -z)
((${#elf[@]} == 5)) || fail "expected 5 ELF files, found ${#elf[@]}: ${elf[*]}"
bash "$here/check_elf_floors.sh" "${GLIBC_FLOOR:?}" "${GLIBCXX_FLOOR:?}" "${CXXABI_FLOOR:?}" "${elf[@]}"
"$stage/cmt/cmt" --help >/dev/null

LC_ALL=C tar --sort=name --format=gnu --mtime="@$SOURCE_DATE_EPOCH" --owner=0 --group=0 --numeric-owner \
	--mode=go-w -C "$stage" -cf - cmt | gzip -9 -n >"$out/$name.tar.gz"
(cd "$out" && sha256sum "$name.tar.gz" | tee "$name.tar.gz.sha256")
