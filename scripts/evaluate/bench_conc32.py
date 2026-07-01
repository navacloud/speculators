#!/usr/bin/env python3
"""Fixed-concurrency (default 32) perf benchmark via guidellm 'concurrent' profile.
Records TTFT, ITL, output tok/s, latency, req/s AND input(prompt)/output token
lengths per subset. Runs INSIDE the spec-moe-eval image (has guidellm).

Usage:
  python3 bench_conc32.py --target http://localhost:8220/v1 --out /path/conc32.csv \
     [--concurrency 32] [--max-requests 128] [--subsets a,b,c]
"""
import argparse, json, os, subprocess, tempfile, csv, statistics, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATASET = "RedHatAI/speculator_benchmarks"
MAPPER = '{"text_column":"prompt"}'
SUBSETS = "HumanEval,math_reasoning,qa,question,rag,summarization,translation"
MAX_TOK = int(os.environ.get("EVAL_MAX_TOKENS", "256"))
MODEL = "google/gemma-4-31B-it"

def warmup(target, n, concurrency, max_tokens=128):
    """Send n concurrent requests to warm CUDA graphs / server before measuring."""
    url = target.rstrip("/") + "/completions"
    prompt = ("Explain in detail how speculative decoding accelerates large "
              "language model inference, step by step. ") * 4
    def _one(_):
        body = json.dumps({"model": MODEL, "prompt": prompt,
                           "max_tokens": max_tokens, "temperature": 0.0}).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=300).read()
            return True
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

def run_subset(target, subset, concurrency, max_requests, tmpdir):
    out = Path(tmpdir) / f"{subset}.json"
    cmd = [
        "guidellm", "benchmark", "--target", target,
        "--data", DATASET, "--data-args", json.dumps({"data_files": f"{subset}.jsonl"}),
        "--data-column-mapper", MAPPER,
        "--profile", "concurrent", "--rate", str(concurrency),
        "--max-requests", str(max_requests),
        "--backend-args", json.dumps({"extras": {"body": {"max_tokens": MAX_TOK}}}),
        "--output-path", str(out),
    ]
    env = os.environ.copy(); env["GUIDELLM__MAX_CONCURRENCY"] = str(concurrency)
    print(f"[conc] {subset}: running concurrency={concurrency} max_requests={max_requests}", flush=True)
    subprocess.run(cmd, check=True, env=env)
    data = json.load(open(out))
    bm = data["benchmarks"][0]
    m = bm.get("metrics", {})
    succ = bm.get("requests", {}).get("successful", [])
    in_tok = [r.get("prompt_tokens") for r in succ if r.get("prompt_tokens") is not None]
    out_tok = [r.get("output_tokens") for r in succ if r.get("output_tokens") is not None]
    return {
        "subset": subset,
        "n_req": len(succ),
        "input_tok_median": round(statistics.median(in_tok), 1) if in_tok else "",
        "output_tok_median": round(statistics.median(out_tok), 1) if out_tok else "",
        "ttft_ms": med(m, "time_to_first_token_ms"),
        "itl_ms": med(m, "inter_token_latency_ms"),
        "output_tps": med(m, "output_tokens_per_second"),
        "latency_s": med(m, "request_latency"),
        "req_per_s": med(m, "requests_per_second", "mean"),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-requests", type=int, default=128)
    ap.add_argument("--warmup-requests", type=int, default=64)
    ap.add_argument("--subsets", default=SUBSETS)
    a = ap.parse_args()
    if a.warmup_requests > 0:
        warmup(a.target, a.warmup_requests, a.concurrency)
    cols = ["subset", "n_req", "input_tok_median", "output_tok_median",
            "ttft_ms", "itl_ms", "output_tps", "latency_s", "req_per_s"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        with tempfile.TemporaryDirectory() as td:
            for s in a.subsets.split(","):
                try:
                    row = run_subset(a.target, s, a.concurrency, a.max_requests, td)
                except Exception as e:
                    print(f"[conc] {s} FAILED: {e}", flush=True)
                    row = {"subset": s, "n_req": 0}
                w.writerow(row); f.flush()
                print(f"[conc] {s} -> {row}", flush=True)
    print(f"[conc] wrote {a.out}", flush=True)

if __name__ == "__main__":
    main()
