from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def run_once(llm, prompt: str, max_tokens: int) -> dict:
    first_token_ms = 0.0
    generated_text = []
    token_chunks = 0
    first = True

    start = time.perf_counter()
    for chunk in llm(prompt, max_tokens=max_tokens, stream=True):
        token = chunk["choices"][0]["text"]
        if token:
            token_chunks += 1
            if first:
                first_token_ms = (time.perf_counter() - start) * 1000.0
                first = False
            generated_text.append(token)
    total_ms = (time.perf_counter() - start) * 1000.0

    return {
        "ttft_ms": first_token_ms,
        "total_ms": total_ms,
        "token_chunks": token_chunks,
        "text_len": len("".join(generated_text)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generation-only llama.cpp microbenchmark (Python)")
    parser.add_argument("--model", required=True, help="Path to GGUF model file")
    parser.add_argument("--prompt", default="Explain in one short paragraph what Rust ownership is.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", default="", help="Optional output JSON file path")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists() or not model_path.is_file():
        raise SystemExit(f"Model not found: {model_path}")

    from llama_cpp import Llama  # type: ignore[import]

    llm = Llama(model_path=str(model_path), n_ctx=args.n_ctx, verbose=False)

    for _ in range(max(0, args.warmup)):
        run_once(llm, args.prompt, args.max_tokens)

    runs = [run_once(llm, args.prompt, args.max_tokens) for _ in range(max(1, args.repeats))]

    ttfts = [r["ttft_ms"] for r in runs]
    totals = [r["total_ms"] for r in runs]
    tok_chunks = [r["token_chunks"] for r in runs]

    out = {
        "language": "python",
        "backend": "llama_cpp",
        "model": str(model_path),
        "n_ctx": args.n_ctx,
        "max_tokens": args.max_tokens,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "prompt_len": len(args.prompt),
        "mean_ttft_ms": statistics.mean(ttfts),
        "p50_ttft_ms": _percentile(ttfts, 0.5),
        "p95_ttft_ms": _percentile(ttfts, 0.95),
        "mean_total_ms": statistics.mean(totals),
        "p50_total_ms": _percentile(totals, 0.5),
        "p95_total_ms": _percentile(totals, 0.95),
        "mean_token_chunks": statistics.mean(tok_chunks),
        "runs": runs,
    }

    text = json.dumps(out, indent=2)
    print(text)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
