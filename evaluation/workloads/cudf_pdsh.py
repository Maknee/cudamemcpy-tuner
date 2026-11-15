"""Run cudf-polars' own PDS-H (TPC-H) benchmark on one GPU with a small device budget, and report query time.

This is `python -m cudf_polars.streaming.benchmarks.pdsh` (its own parser and runner, streaming engine, SPMD
frontend, RapidsMPF spilling), run in this process so `cmt trace` sees it. Queries 1,3,9,13,18,21 at scale 10 with a
2 GiB device limit, so they spill (the setting measured on 2026-11-02).
Copies: parquet input CPU->GPU and spill/unspill both ways. By default RapidsMPF spills into a pinned host
pool. That run measured 13 GB CPU->GPU per pass; with --no-pinned-memory
(passed through here) 48% of query time was copy. Most spill copies use cudaMemcpyBatchAsync, which neither the hook nor the
trace covers (both document batch APIs as excluded).
Metric: sum over queries of the last iteration's duration (the first iteration also compiles).
"""
import argparse
import json
import os
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=os.path.join(os.environ.get("CMT_ENV_ROOT", "tmp/envs"), "data/pdsh-sf10"))
    parser.add_argument("--queries", default="1,3,9,13,18,21")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--spill-device-limit", default="2GiB")
    args, passthrough = parser.parse_known_args()

    from cudf_polars.streaming.benchmarks.pdsh import PDSHQueries, build_parser, parse_args, run_polars

    with tempfile.TemporaryDirectory(prefix="cmt-pdsh-") as work:
        output = os.path.join(work, "pdsh.json")
        options = parse_args([args.queries, "--frontend", "spmd", "--path", args.data, "--scale", "10",
                              "--iterations", str(args.iterations), "--spill-device-limit", args.spill_device_limit,
                              "-o", output, "--no-print-results", "--no-summarize", *passthrough],
                             parser=build_parser(num_queries=22))
        # The same thread defaults as pdsh's __main__ block for GPU frontends.
        os.environ.setdefault("POLARS_MAX_THREADS", "1")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        try:
            run_polars(PDSHQueries, options)
        except SystemExit as stop:  # pdsh exits with its status code: 0 = all queries succeeded
            if stop.code not in (0, None):
                raise
        options.output.close()  # pdsh leaves its -o file open and unflushed
        with open(output) as file:
            report = json.load(file)
    last = {query: runs[-1] for query, runs in report["records"].items()}
    failed = [query for query, run in last.items() if run["status"] != "success"]
    if failed:
        raise SystemExit(f"cudf_pdsh.py: queries failed: {failed}")
    print(json.dumps({"metric": "query_seconds", "value": sum(run["duration"] for run in last.values()), "unit": "s",
                      "per_query": {query: run["duration"] for query, run in last.items()},
                      "first_iteration": {query: runs[0]["duration"] for query, runs in report["records"].items()},
                      "spill_device_limit": args.spill_device_limit, "versions": report["versions"]}))


if __name__ == "__main__":
    main()
