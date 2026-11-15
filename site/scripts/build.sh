#!/usr/bin/env bash
# Build browser assets and export a bounded, entirely static snapshot.
set -euo pipefail
cd "$(dirname "$0")/.."
DX=${DX:-dx}
"$DX" --version | grep -Eq '^dioxus 0\.7\.10([[:space:]]|$)' || {
  echo "Need dioxus-cli 0.7.10; install with cargo install dioxus-cli --version 0.7.10 --locked" >&2
  exit 1
}
cargo build --release --locked --features export
# Dioxus retains old hashed assets between builds; publish only this build's files.
rm -rf -- target/frontend target/dx/cudamemcpy-site/release/web/public
# Panic locations can embed absolute paths even without debug symbols.
RUSTFLAGS="${RUSTFLAGS:-} --remap-path-prefix=$HOME=/build/home --remap-path-prefix=$PWD=/build/site --remap-path-prefix=${CARGO_HOME:-$HOME/.cargo}=/build/cargo --remap-path-prefix=${RUSTUP_HOME:-$HOME/.rustup}=/build/rustup" \
  "$DX" bundle --locked --web --release --debug-symbols false --out-dir target/frontend "$@"
export CMX_DB=${CMX_DB:-$PWD/data/cmx.duckdb}
target/release/cudamemcpy-site build --web target/frontend/public --out dist
