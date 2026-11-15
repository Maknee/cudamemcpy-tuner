# Static benchmark website

The existing benchmark explorer, exported as static files for Cloudflare Pages.
Dioxus runs in the browser; Rust and DuckDB are used only to prepare the numbers.
Filtering, charts, pagination and JSON/CSV downloads work in the browser.
The header links to GitHub and the tune pull-request instructions. Change
`GITHUB_REPOSITORY` in `src/main.rs` if the repository moves; both links follow it.

## Build and preview

Requires Rust 1.89+, a C++ compiler for bundled DuckDB, Python 3, and Dioxus CLI
0.7.10. Build independently of CMake:

```sh
cd site
rustup target add wasm32-unknown-unknown
cargo install dioxus-cli --version 0.7.10 --locked
cargo build --release --locked --features export

cp runs.example.tsv runs.tsv
# Edit runs.tsv to list the local tune.json inputs and their labels.
python3 scripts/import.py runs.tsv
scripts/build.sh
scripts/run.sh
```

Open `http://127.0.0.1:8787`. Both exporter and browser use release builds. The first
build takes a few minutes. `DX` overrides the Dioxus CLI executable; `CMX_DB`
selects the input database (default `site/data/cmx.duckdb`). `PORT` changes the
preview port. Shell scripts resolve relative paths from `site/`.
The browser build remaps local source/dependency paths to neutral `/build/` paths
and omits debug symbols so the WASM does not expose the builder's home directory.

Upload **the contents of `site/dist/`** to Cloudflare Pages. No server, database,
Worker or Functions are deployed. See
[Cloudflare's Direct Upload instructions](https://developers.cloudflare.com/pages/get-started/direct-upload/).
The site is served at the domain root, with static pages at `/` and `/run/{id}/`.
`cmt run --remote-config=https://YOUR-DOMAIN -- APP` reads `data/summary.json`,
chooses a run matching the CPU/GPU models, and downloads `data/{id}/tune.json`.
These are ordinary static files; no API service is needed. A specific tune URL also
works with `cmt run --config URL -- APP`.

Every build checks the final files against the project budgets: **under 8,000,000
bytes total** and **under 2,000,000 bytes per file**, before HTTP compression.
`sizes.json` lists file sizes. A failed build leaves the previous output intact.

## Update the numbers

Run `CMT_INTERNAL_ENABLE=1 cmt check --write` on local tune files before importing them. Input paths in
`runs.tsv` are relative to that manifest, which lists every run wanted in the
snapshot. A successful import replaces the database; failed imports preserve it.
Rebuild and upload `dist/` to update the website. Local inputs, database and build
outputs are ignored by Git.

Conversion keeps complete, byte-verified copy measurements. It records omitted
rows in the run note. Curves prefer the tuner's selected config, falling back to
the fastest retained measurement. The site displays copy results, not allocation
results. Verification badges reflect data plausibility, not reproduced performance:
NC indicates invalid values or a PCIe ceiling violation; T0 indicates partial data
or warnings; T1 indicates no disqualifying issue was found.

Point files use shared dictionaries to fit the budget without dropping numbers.
JSON downloads retain the original site's rounding and can be imported again;
CSV retains stored numeric precision. Machine/selection fields are allowlisted,
with heuristic filtering for other strings; review inputs before publishing.

## Layout and checks

`pages/` and `chart.rs` render the existing UI. `client.rs` loads static data and
handles filtering/downloads. `backend/` contains the offline importer and exporter;
`publish.rs` writes the static directory and checks its size.

For website changes, run the release build, preview it with `scripts/run.sh`, and
check the charts, filters and downloads in a browser. No GPU is needed.
