"""
Extract QA pairs from HPC-AgentBench's canonical.jsonl.

Keeps records that are valid and have a real reference reply, then applies
length filters to drop pairs too short to be substantive.

Usage:
    python extract_pairs.py ../hpc_agentbench/pipeline_outputs/canonical.jsonl qa_pairs.json
"""

import argparse
import json

MIN_QUESTION_WORDS = 10
MIN_ANSWER_WORDS = 30


def word_count(text: str) -> int:
    return len(text.split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("canonical_path")
    parser.add_argument("output_path")
    parser.add_argument("--min-question-words", type=int, default=MIN_QUESTION_WORDS)
    parser.add_argument("--min-answer-words", type=int, default=MIN_ANSWER_WORDS)
    args = parser.parse_args()

    with open(args.canonical_path) as f:
        records = [json.loads(line) for line in f]

    dropped_invalid = 0
    dropped_no_reply = 0
    dropped_short_question = 0
    dropped_short_answer = 0
    pairs = []

    for r in records:
        if not r.get("is_valid"):
            dropped_invalid += 1
            continue

        evaluation = r.get("evaluation", {})
        if not evaluation.get("has_reference_admin_reply"):
            dropped_no_reply += 1
            continue

        question = r.get("instruction", "")
        answer = evaluation.get("reference_admin_reply", "")

        if word_count(question) < args.min_question_words:
            dropped_short_question += 1
            continue
        if word_count(answer) < args.min_answer_words:
            dropped_short_answer += 1
            continue

        pairs.append({
            "id": r["source_ticket_id"],
            "question": question,
            "answer": answer,
            "tier": r.get("release_tier"),
        })

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    print(f"Records in canonical.jsonl:   {len(records)}")
    print(f"Dropped (is_valid false):     {dropped_invalid}")
    print(f"Dropped (no reference reply): {dropped_no_reply}")
    print(f"Dropped (question < {args.min_question_words} words):  {dropped_short_question}")
    print(f"Dropped (answer < {args.min_answer_words} words):    {dropped_short_answer}")
    print(f"Pairs written:                {len(pairs)} -> {args.output_path}")


if __name__ == "__main__":
    main()
