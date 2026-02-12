#!/usr/bin/env python3
"""Minimal eval CLI. Usage: PYTHONPATH=src python scripts/eval_domain.py --domain_id iam --limit 20"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Force mock mode before any imports
os.environ["USE_MOCK_LLM"] = "true"

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.app.domains.registry import get_domain_registry


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain_id", required=True)
    p.add_argument("--limit", type=int, default=20)
    args = p.parse_args()

    eval_path = repo_root / "eval" / f"{args.domain_id}.jsonl"
    if not eval_path.exists():
        print(f"Error: {eval_path} not found", file=sys.stderr)
        sys.exit(1)

    registry = get_domain_registry()
    adapter = registry.get_adapter(args.domain_id)
    if not adapter:
        print(f"Error: domain '{args.domain_id}' not found", file=sys.stderr)
        sys.exit(1)

    lines = []
    with open(eval_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lines.append(json.loads(line))
            if len(lines) >= args.limit:
                break

    total = len(lines)
    refusals = 0
    citations_present = 0
    should_refuse_correct = 0
    should_refuse_total = 0
    latencies_ms = []

    for row in lines:
        q = row.get("question", "")
        expect_refuse = row.get("should_refuse", False)
        t0 = time.perf_counter()
        out = adapter.ask(q)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(elapsed_ms)

        is_refusal = False
        if isinstance(out, dict):
            ans = out.get("answer", "")
            cites = out.get("citations", [])
            if "don't know" in ans.lower() or "do not know" in ans.lower():
                refusals += 1
                is_refusal = True
            if cites:
                citations_present += 1
        else:
            if "don't know" in str(out).lower():
                refusals += 1
                is_refusal = True

        if expect_refuse:
            should_refuse_total += 1
            if is_refusal:
                should_refuse_correct += 1

    refusal_rate = refusals / total if total else 0
    citation_present_rate = citations_present / total if total else 0
    should_refuse_accuracy = should_refuse_correct / should_refuse_total if should_refuse_total else 0
    avg_latency_ms = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0

    print(f"total: {total}")
    print(f"refusal_rate: {refusal_rate:.2%}")
    print(f"citation_present_rate: {citation_present_rate:.2%}")
    print(f"should_refuse_accuracy: {should_refuse_accuracy:.2%}" if should_refuse_total else "should_refuse_accuracy: n/a (no should_refuse rows)")
    print(f"avg_latency_ms: {avg_latency_ms:.0f}")


if __name__ == "__main__":
    main()
