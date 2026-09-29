"""
Two-pass selection of the final N QA pairs.

Pass 1: take grounded + is_valid pairs (per HPC-AgentBench's canonical.jsonl)
        that survive all judge filters, ranked by RAG value.
Pass 2: if Pass 1 comes up short of N, backfill from the remaining
        (reconstructed) pairs, again ranked by RAG value.

Question and answer text is copied through untouched.

Usage:
    python select_final.py qa_pairs_judged.jsonl pipeline_outputs/canonical.jsonl \
        top_250_qa.json --n 250 --max-per-category 60 --dup-threshold 0.6
"""

import argparse
import json
from collections import defaultdict

CATEGORY_MERGES = {
    "resource_cpu_gpu_memory": "slurm_and_resources",
    "slurm_job_scheduling": "slurm_and_resources",
}


def effective_category(judge: dict) -> str:
    cat = judge.get("topic_category", "other")
    return CATEGORY_MERGES.get(cat, cat)


def jaccard_similarity(a: str, b: str) -> float:
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def is_near_duplicate(record: dict, kept: list, threshold: float) -> bool:
    for k in kept:
        if jaccard_similarity(record["question"], k["question"]) >= threshold:
            return True
    return False


def passes_judge_filters(judge: dict) -> bool:
    if judge.get("judge_parse_error") or judge.get("judge_call_error"):
        return False
    return bool(
        judge.get("has_clear_qa_pair")
        and judge.get("is_customer_facing")
        and judge.get("is_hpc_related")
    )


def select(pool: list, n_wanted: int, max_per_category: int, dup_threshold: float,
           already_selected: list, category_counts: defaultdict) -> list:
    # take the highest-scoring pairs from pool, skipping
    # near-duplicates of anything already chosen and respecting category caps.
    
    pool = sorted(pool, key=lambda r: r["judge"].get("rag_value_score", 0), reverse=True)
    chosen = []
    for r in pool:
        if len(chosen) >= n_wanted:
            break
        cat = effective_category(r["judge"])
        if category_counts[cat] >= max_per_category:
            continue
        if is_near_duplicate(r, already_selected + chosen, dup_threshold):
            continue
        chosen.append(r)
        category_counts[cat] += 1
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("judged_path")
    parser.add_argument("canonical_path")
    parser.add_argument("output_path")
    parser.add_argument("--n", type=int, default=250)
    parser.add_argument("--max-per-category", type=int, default=60)
    parser.add_argument("--dup-threshold", type=float, default=0.6)
    args = parser.parse_args()

    judged = [json.loads(l) for l in open(args.judged_path)]

    canonical = {}
    with open(args.canonical_path) as f:
        for line in f:
            r = json.loads(line)
            canonical[r["source_ticket_id"]] = r

    grounded_valid_ids = {
        tid for tid, r in canonical.items()
        if r.get("release_tier") == "grounded" and r.get("is_valid")
    }

    eligible = [r for r in judged if passes_judge_filters(r.get("judge", {}))]
    excluded = len(judged) - len(eligible)

    grounded_pool = [r for r in eligible if r["id"] in grounded_valid_ids]
    other_pool = [r for r in eligible if r["id"] not in grounded_valid_ids]

    category_counts = defaultdict(int)

    # Pass 1: grounded + valid
    pass1 = select(grounded_pool, args.n, args.max_per_category,
                   args.dup_threshold, [], category_counts)
    for r in pass1:
        r["selection_tier"] = "grounded_valid"

    # Pass 2: backfill
    remaining = args.n - len(pass1)
    pass2 = []
    if remaining > 0:
        pass2 = select(other_pool, remaining, args.max_per_category,
                       args.dup_threshold, pass1, category_counts)
        for r in pass2:
            r["selection_tier"] = "backfill_reconstructed"

    selected = pass1 + pass2

    output = [
        {
            "id": r["id"],
            "question": r["question"],
            "answer": r["answer"],
            "tier": r.get("tier"),
            "selection_tier": r["selection_tier"],
            "judge": r["judge"],
        }
        for r in selected
    ]

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Total judged: {len(judged)}")
    print(f"Excluded by judge filters: {excluded}")
    print(f"Eligible pool: {len(eligible)} "
          f"({len(grounded_pool)} grounded+valid, {len(other_pool)} other)")
    print()
    print(f"Pass 1 (grounded+valid): {len(pass1)}")
    print(f"Pass 2 (backfill):       {len(pass2)}")
    print(f"Total selected:          {len(selected)}")
    print()
    print("Final selection by category:")
    for cat, count in sorted(category_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {cat}: {count}")
    print(f"\nWrote {args.output_path}")


if __name__ == "__main__":
    main()
