#!/usr/bin/env python3
"""
Model evaluation: per-domain benchmarks across Grok variants + non-RAG baseline.

Usage:
  PYTHONPATH=src python scripts/eval_models.py [--domains iam,security] [--mock]

Requires GROK_API_KEY when not using --mock.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Model variants: 3 Grok models + non-RAG baseline
RAG_MODELS = ["grok-4-latest", "grok-4-fast-reasoning", "grok-3-mini"]
NON_RAG_BASELINE = "non-rag (grok-4-latest)"


def _is_refusal(out) -> bool:
    """Check if response is a refusal."""
    if isinstance(out, dict):
        ans = out.get("answer", "")
    else:
        ans = str(out)
    # Handle raw JSON string (non-RAG returns JSON as text)
    if isinstance(ans, str) and ans.strip().startswith("{"):
        try:
            parsed = json.loads(ans)
            ans = parsed.get("answer", ans)
        except json.JSONDecodeError:
            pass
    text = str(ans).lower().replace("\u2019", "'")
    return "don't know" in text or "do not know" in text


def _run_eval(adapter, lines: list, model_override: str | None, use_rag: bool) -> dict:
    """Run eval on adapter, return metrics."""
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
        out = adapter.ask(
            q,
            include_rag_context=use_rag,
            model_override=model_override,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(elapsed_ms)

        # Normalize non-RAG response to dict-like for metrics
        if not isinstance(out, dict):
            out = {"answer": str(out), "confidence": "n/a", "citations": []}

        if _is_refusal(out):
            refusals += 1
        if out.get("citations"):
            citations_present += 1

        if expect_refuse:
            should_refuse_total += 1
            if _is_refusal(out):
                should_refuse_correct += 1

    refusal_rate = refusals / total if total else 0
    citation_present_rate = citations_present / total if total else 0
    should_refuse_accuracy = (
        should_refuse_correct / should_refuse_total if should_refuse_total else 0
    )
    avg_latency_ms = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0

    return {
        "total": total,
        "refusal_rate": refusal_rate,
        "citation_present_rate": citation_present_rate,
        "should_refuse_accuracy": should_refuse_accuracy,
        "should_refuse_total": should_refuse_total,
        "avg_latency_ms": avg_latency_ms,
    }


def main():
    p = argparse.ArgumentParser(description="Per-domain model benchmarks")
    p.add_argument(
        "--domains",
        type=str,
        default=None,
        help="Comma-separated domain_ids (default: all with eval files)",
    )
    p.add_argument(
        "--mock",
        action="store_true",
        help="Use mock LLM (no API calls)",
    )
    p.add_argument("--limit", type=int, default=20, help="Max questions per domain")
    args = p.parse_args()

    if args.mock:
        os.environ["USE_MOCK_LLM"] = "true"

    from src.app.core import GrokClient, RAGEngine, load_domain_configs
    from src.app.domains.adapter import DomainAdapter

    configs_base = repo_root / "configs"
    configs = load_domain_configs(configs_base)
    configs_by_id = {c.domain_id: c for c in configs}

    eval_dir = repo_root / "eval"
    if args.domains:
        domain_ids = [d.strip() for d in args.domains.split(",")]
    else:
        domain_ids = [
            p.stem for p in eval_dir.glob("*.jsonl") if configs_by_id.get(p.stem)
        ]

    if not domain_ids:
        print("No domains to evaluate. Add eval/<domain_id>.jsonl and configs.", file=sys.stderr)
        sys.exit(1)

    grok = GrokClient()
    rag = RAGEngine(configs_base=configs_base)

    results: dict[str, dict[str, dict]] = {}  # domain_id -> variant -> metrics

    for domain_id in domain_ids:
        config = configs_by_id.get(domain_id)
        if not config:
            print(f"Skipping {domain_id}: no config", file=sys.stderr)
            continue

        eval_path = eval_dir / f"{domain_id}.jsonl"
        if not eval_path.exists():
            print(f"Skipping {domain_id}: no {eval_path}", file=sys.stderr)
            continue

        lines = []
        with open(eval_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                lines.append(json.loads(line))
                if len(lines) >= args.limit:
                    break

        if not lines:
            print(f"Skipping {domain_id}: no questions", file=sys.stderr)
            continue

        adapter = DomainAdapter(config=config, grok_client=grok, rag_engine=rag)
        results[domain_id] = {}

        # RAG variants
        for model in RAG_MODELS:
            metrics = _run_eval(adapter, lines, model_override=model, use_rag=True)
            results[domain_id][model] = metrics

        # Non-RAG baseline
        metrics = _run_eval(
            adapter, lines, model_override="grok-4-latest", use_rag=False
        )
        results[domain_id][NON_RAG_BASELINE] = metrics

    # Print table
    print("\n" + "=" * 80)
    print("MODEL EVALUATION: per-domain benchmarks")
    print("=" * 80)

    for domain_id, variants in results.items():
        print(f"\n--- {domain_id} ---")
        print(f"{'variant':<35} {'refusal':>8} {'citation':>8} {'sr_acc':>8} {'latency_ms':>10}")
        print("-" * 75)
        for variant, m in variants.items():
            sr = f"{m['should_refuse_accuracy']:.0%}" if m["should_refuse_total"] else "n/a"
            print(
                f"{variant:<35} {m['refusal_rate']:>7.0%} {m['citation_present_rate']:>7.0%} "
                f"{sr:>8} {m['avg_latency_ms']:>10.0f}"
            )

    print("\n" + "=" * 80)
    if args.mock:
        print("(mock mode - no API calls)")
    print()


if __name__ == "__main__":
    main()
