#!/usr/bin/env python3
"""Fixed-concurrency latency benchmark on a SINGLE proper dataset (default GSM8K
test, 1319 unique prompts) via guidellm 'concurrent' profile. No prompt cycling
(caps at unique prompts) so prefix-caching does not skew latency.

Records TTFT, ITL, output tok/s, latency, req/s + input/output token lengths.
Runs INSIDE the spec-moe-eval image (has guidellm).

Usage:
  python3 bench_gsm8k.py --target http://localhost:8220/v1 --out /path/out.csv \
     --label base_gsm8k [--concurrency 32] [--max-requests 1000] \
     [--dataset openai/gsm8k] [--data-args '{"name":"main","split":"test"}'] \
     [--column question] [--warmup-requests 64]
"""
import argparse, json, os, subprocess, tempfile, csv, statistics, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MAX_TOK = int(os.environ.get("EVAL_MAX_TOKENS", "512"))
MODEL = "google/gemma-4-31B-it"

def warmup(target, n, concurrency, max_tokens=128):
    url = target.rstrip("/") + "/completions"
    prompt = ("Solve the following math problem step by step, showing your "
              "reasoning clearly before the final answer. ") * 3
    def _one(_):
        body = json.dumps({"model": MODEL, "prompt": prompt,
                           "max_tokens": max_tokens, "temperature": 0.0}).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=300).read(); return True
        except Exception as e:
            print(f"[warmup] req failed: {e}", flush=True); return False
    print(f"[warmup] sending {n} requests @ concurrency {concurrency}...", flush=True)
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        ok = sum(ex.map(_one, range(n)))
    print(f"[warmup] done ({ok}/{n} ok)", flush=True)

def med(metrics, key, stat="median"):
    try:
        return round(float(metrics.get(key, {}).get("successful", {}).get(stat, "nan")), 3)
    except Exception:
        return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="gsm8k")
    ap.add_argument("--dataset", default="openai/gsm8k")
    ap.add_argument("--data-args", default='{"name":"main","split":"test"}')
    ap.add_argument("--column", default="question")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-requests", type=int, default=1000)
    ap.add_argument("--warmup-requests", type=int, default=64)
    a = ap.parse_args()

    if a.warmup_requests > 0:
        warmup(a.target, a.warmup_requests, a.concurrency)

    cols = ["label", "n_req", "input_tok_median", "output_tok_mean",
            "ttft_ms", "itl_ms", "output_tps_agg", "latency_s", "req_per_s"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "bm.json"
        cmd = [
            "guidellm", "benchmark", "--target", a.target,
            "--data", a.dataset, "--data-args", a.data_args,
            "--data-column-mapper", json.dumps({"text_column": a.column}),
            "--profile", "concurrent", "--rate", str(a.concurrency),
            "--max-requests", str(a.max_requests),
            "--backend-args", json.dumps({"extras": {"body": {"max_tokens": MAX_TOK}}}),
            "--output-path", str(out),
        ]
        env = os.environ.copy(); env["GUIDELLM__MAX_CONCURRENCY"] = str(a.concurrency)
        print(f"[gsm8k] {a.label}: dataset={a.dataset} args={a.data_args} col={a.column} "
              f"conc={a.concurrency} max_req={a.max_requests} max_tok={MAX_TOK}", flush=True)
        subprocess.run(cmd, check=True, env=env)
        data = json.load(open(out)); bm = data["benchmarks"][0]; m = bm.get("metrics", {})
        succ = bm.get("requests", {}).get("successful", [])
        in_tok = [r.get("prompt_tokens") for r in succ if r.get("prompt_tokens") is not None]
        out_tok = [r.get("output_tokens") for r in succ if r.get("output_tokens") is not None]
        # Aggregate (system) output throughput = system req/s (mean) * mean output tokens/req.
        # NOTE: do NOT use median of output_tokens_per_second — that is a per-interval rate
        # distribution whose median badly understates the true aggregate.
        rps_mean = med(m, "requests_per_second", "mean")
        out_tok_mean = round(statistics.mean(out_tok), 1) if out_tok else ""
        try:
            output_tps_agg = round(float(rps_mean) * statistics.mean(out_tok), 1) if out_tok else ""
        except Exception:
            output_tps_agg = ""
        row = {
            "label": a.label, "n_req": len(succ),
            "input_tok_median": round(statistics.median(in_tok), 1) if in_tok else "",
            "output_tok_mean": out_tok_mean,
            "ttft_ms": med(m, "time_to_first_token_ms"),
            "itl_ms": med(m, "inter_token_latency_ms"),
            "output_tps_agg": output_tps_agg,
            "latency_s": med(m, "request_latency"),
            "req_per_s": rps_mean,
        }
    write_header = not Path(a.out).exists()
    with open(a.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if write_header: w.writeheader()
        w.writerow(row)
    print(f"[gsm8k] {a.label} -> {row}", flush=True)
    print(f"[gsm8k] wrote {a.out}", flush=True)

if __name__ == "__main__":
    main()
