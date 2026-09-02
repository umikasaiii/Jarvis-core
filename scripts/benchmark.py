#!/usr/bin/env python3
"""Benchmark JARVIS Core's streaming endpoint against a running server.

Sends a set of standard prompts to /v1/ai/stream and reports, per request:

    Model, Backend, Prompt tokens (approx.), Generated tokens,
    TTFT (time to first token), tokens/sec, Total latency, RAM usage (best-effort)

Usage:
    python scripts/benchmark.py --base-url http://127.0.0.1:8000 --target FAST
    python scripts/benchmark.py --target BRAIN --prompt "Explain quicksort briefly."
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from dataclasses import dataclass

import httpx

DEFAULT_PROMPTS = [
    "Ciao, che ore sono?",
    "Riassumi in tre punti i vantaggi di un server AI locale.",
    "Spiega in dettaglio come funziona il routing FAST/BRAIN e perché è utile.",
]


def _approx_tokens(text: str) -> int:
    # Rough approximation (no tokenizer dependency): ~4 chars/token.
    return max(1, len(text) // 4)


def _rss_mb() -> float | None:
    try:
        import psutil  # optional dependency, not in requirements.txt

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return None


@dataclass
class BenchmarkResult:
    prompt: str
    model_used: str | None
    target_used: str | None
    prompt_tokens_approx: int
    generated_tokens: int
    ttft_ms: float | None
    total_latency_ms: float
    tokens_per_sec: float | None
    rss_mb: float | None
    error: str | None = None


def run_one(client: httpx.Client, base_url: str, prompt: str, target: str) -> BenchmarkResult:
    payload = {
        "requestId": str(uuid.uuid4()),
        "protocolVersion": "1",
        "text": prompt,
        "preferredTarget": target,
    }
    start = time.perf_counter()
    ttft = None
    generated = 0
    model_used = None
    target_used = None
    error = None

    try:
        with client.stream("POST", f"{base_url}/v1/ai/stream", json=payload, timeout=120) as resp:
            resp.raise_for_status()
            buffer = ""
            for chunk in resp.iter_text():
                buffer += chunk
                while "\n\n" in buffer:
                    raw, buffer = buffer.split("\n\n", 1)
                    if not raw.startswith("data: "):
                        continue
                    event = json.loads(raw.removeprefix("data: "))
                    if event["type"] == "token":
                        if ttft is None:
                            ttft = (time.perf_counter() - start) * 1000
                        generated += 1
                    elif event["type"] == "done":
                        model_used = event.get("modelUsed")
                        target_used = event.get("targetUsed")
                        generated = event.get("tokensGenerated", generated)
                    elif event["type"] == "error":
                        error = event.get("error")
    except httpx.HTTPError as exc:
        error = str(exc)

    total_ms = (time.perf_counter() - start) * 1000
    tokens_per_sec = (generated / (total_ms / 1000)) if total_ms > 0 and generated else None

    return BenchmarkResult(
        prompt=prompt,
        model_used=model_used,
        target_used=target_used,
        prompt_tokens_approx=_approx_tokens(prompt),
        generated_tokens=generated,
        ttft_ms=ttft,
        total_latency_ms=total_ms,
        tokens_per_sec=tokens_per_sec,
        rss_mb=_rss_mb(),
        error=error,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--target", default="AUTO", choices=["AUTO", "FAST", "BRAIN"])
    parser.add_argument("--prompt", action="append", help="Custom prompt (repeatable)")
    args = parser.parse_args()

    prompts = args.prompt or DEFAULT_PROMPTS

    results: list[BenchmarkResult] = []
    with httpx.Client() as client:
        for prompt in prompts:
            results.append(run_one(client, args.base_url.rstrip("/"), prompt, args.target))

    header = (
        f"{'Model':<16} {'Target':<7} {'PromptTok':>9} {'GenTok':>7} "
        f"{'TTFT(ms)':>9} {'tok/s':>7} {'Total(ms)':>10} {'RAM(MB)':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        if r.error:
            print(f"ERROR: {r.prompt[:40]!r} -> {r.error}")
            continue
        print(
            f"{(r.model_used or '?'):<16} {(r.target_used or '?'):<7} "
            f"{r.prompt_tokens_approx:>9} {r.generated_tokens:>7} "
            f"{(r.ttft_ms or 0):>9.1f} {(r.tokens_per_sec or 0):>7.1f} "
            f"{r.total_latency_ms:>10.1f} {(r.rss_mb or 0):>8.1f}"
        )


if __name__ == "__main__":
    main()
