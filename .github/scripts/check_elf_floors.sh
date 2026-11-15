#!/usr/bin/env bash
# Check that ELF files need no newer glibc, libstdc++ or C++ ABI symbol versions than the stated floors.
# Usage: check_elf_floors.sh GLIBC_MAX GLIBCXX_MAX CXXABI_MAX FILE...
#   e.g. check_elf_floors.sh 2.35 3.4.29 1.3.13 dist/cmt/cmt dist/cmt/libcmt_hook.so
# Reads each file's version references (objdump -p, "Version References"): the versions the dynamic loader
# requires of libc, libstdc++ and friends, which is what decides the oldest system a binary starts on. Prints the
# highest of each per file and fails (exit 1) if any is above its floor.
set -euo pipefail

if (($# < 4)); then
	echo "usage: check_elf_floors.sh GLIBC_MAX GLIBCXX_MAX CXXABI_MAX FILE..." >&2
	exit 2
fi
declare -A floor=([GLIBC]=$1 [GLIBCXX]=$2 [CXXABI]=$3)
shift 3

# True when version $1 is above version $2.
above() {
	[[ $1 != "$2" && $(printf '%s\n%s\n' "$1" "$2" | sort -V | tail -n 1) == "$1" ]]
}

status=0
for file in "$@"; do
	refs=$(objdump -p "$file" | awk '/^Version References:/ { on = 1; next } on && /^$/ { exit } on')
	line="$file:"
	for kind in GLIBC GLIBCXX CXXABI; do
		highest=$(grep -oE " ${kind}_[0-9]+(\.[0-9]+)*\$" <<<"$refs" | sed "s/ ${kind}_//" | sort -Vu | tail -n 1 || true)
		line+=" $kind ${highest:-none}"
		if [[ -n $highest ]] && above "$highest" "${floor[$kind]}"; then
			echo "::error file=$file::needs ${kind}_$highest, above the floor ${kind}_${floor[$kind]}"
			status=1
		fi
	done
	echo "$line"
done
exit "$status"
