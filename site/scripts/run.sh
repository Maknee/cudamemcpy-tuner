#!/usr/bin/env bash
# Preview exactly the static directory uploaded to Cloudflare Pages.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f dist/index.html ] || { echo "Build first with scripts/build.sh." >&2; exit 1; }
exec python3 -m http.server "${PORT:-8787}" --bind "${IP:-127.0.0.1}" --directory dist
