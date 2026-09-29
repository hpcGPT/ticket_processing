"""
LLM-as-judge over the pairs

Input:  qa_pairs.json (list of {id, question, answer, tier})
Output: qa_pairs_judged.jsonl (one JSON object per line: original pair +
        judge output)

Usage:
    python judge_pairs.py qa_pairs.json qa_pairs_judged.jsonl \
        --server-base-url http://localhost:8000/v1 --model <model>
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

TOPIC_CATEGORIES = [
    "slurm_job_scheduling",
    "resource_cpu_gpu_memory",
    "storage_filesystem",
    "ondemand",
    "software_modules_environment",
    "networking_connectivity",
    "account_project_admin",
    "other",
]

JUDGE_SYSTEM_PROMPT = f"""\
You are evaluating a candidate question/answer pair extracted from a real \
HPC center support ticket, for inclusion in a curated RAG reference dataset \
for an HPC helpdesk assistant. You are a JUDGE only -- you must NOT rewrite, \
summarize, paraphrase, or otherwise alter the question or answer text. \
Evaluate them exactly as given.

Respond with ONLY a single JSON object, no other text, no markdown fences:

{{
  "has_clear_qa_pair": true or false,
  "has_clear_qa_pair_evidence": "<one short phrase justifying this>",
    // true ONLY if the answer represents a single, well-defined resolution
    // to the question -- a real explanation, fix, or direct response.
    // Mark FALSE if:
    //   - the answer is actually a jumbled multi-turn conversation thread
    //     (multiple back-and-forth exchanges, follow-up questions, and
    //     replies all mixed together) rather than one clear answer
    //   - the answer is only future planning/logistics (e.g. scheduling a
    //     call, "does this time work?") with no actual technical answer
    //   - the answer is only a greeting, acknowledgment, or "thanks!" with
    //     no substantive content
    //   - the content is a vague description or suggestion rather than an
    //     actual resolution to the stated problem

  "is_customer_facing": true or false,
  "is_customer_facing_evidence": "<quote proving this, or empty if true and obvious>",
    // Mark FALSE only if the ENTIRE answer consists of internal notes/
    // records with NO part written to the customer (e.g. a pure
    // third-person log like "Responded to X on date Y" with nothing else).
    // If ANY part of the answer is a message actually addressed to or
    // written for the customer (a greeting, a direct explanation, a reply
    // signed off to them), mark this TRUE even if other parts of the
    // answer are terse, administrative, or automated. When uncertain,
    // mark TRUE (do not default to excluding).

  "is_hpc_related": true or false,
    // true if the question concerns actually using HPC resources: job
    // scheduling, software/modules, storage, compute environment,
    // performance, connectivity to the cluster, etc. False only for
    // things with no real HPC/technical-computing content (e.g. pure
    // billing paperwork, unrelated personal requests).

  "topic_category": one of {TOPIC_CATEGORIES},
    // Pick the single best-fitting category for the question's subject.

  "rag_value_score": <integer 0-10>,
    // How valuable is this EXACT pair as a standalone reference document
    // for helping future users with similar problems? Consider as
    // factors:
    //   - Defined: Is there a clearly defined question and answer pair in the ticket, 
    //     rather than merely suggestions or descriptions? 
    //   - Reusability: would another HPC user genuinely benefit from this?
    //   - Commonness: is this a problem type that recurs across users?
    //   - Actionability: concrete commands, code, config, or diagnostic
    //     steps included?
    //   - Root cause: does the answer explain WHY the problem happens?
    //   - Technical content and depth
    //   - Generalizability: useful beyond this one specific ticket/project?
    //   - Clarity
    //   - Question specificity: does the question read like a real support
    //     ticket with concrete technical detail (an error message, exit
    //     code, error logs, terminal commands, failure descriptions), which makes
    //     it a better retrieval trigger for similar future problems, versus
    //     a vague non-technical question?
    // A vague or poorly-described question can still score high if the
    // scenario itself is common and the answer teaches something reusable
    // (e.g. a beginner's vague error report is itself a common support
    // scenario). Conversely, a well-formed but narrowly administrative
    // exchange (e.g. disabling one specific project account) should score
    // LOW even though it's perfectly legitimate and customer-facing --
    // score for REUSABLE HPC KNOWLEDGE VALUE, not for "was this a good,
    // well-handled ticket."

  "rag_value_reasoning": "<one brief sentence justifying the score>"
}}
"""


def build_client(server_base_url: str, api_key: str = "EMPTY") -> OpenAI:
    return OpenAI(base_url=server_base_url, api_key=api_key)


def judge_pair(client: OpenAI, model: str, pair: dict) -> dict:
    user_content = f"QUESTION:\n{pair['question']}\n\nANSWER:\n{pair['answer']}"
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.0,
        max_tokens=350,
    )
    raw = (response.choices[0].message.content or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    try:
        judged = json.loads(raw)
    except json.JSONDecodeError:
        judged = {"judge_parse_error": True, "raw_response": raw}
    return {**pair, "judge": judged}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("--server-base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()

    pairs = json.load(open(args.input_path))
    client = build_client(args.server_base_url)

    results = []
    errors = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(judge_pair, client, args.model, p): p for p in pairs}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                results.append(future.result())
            except Exception as exc:
                errors += 1
                pair = futures[future]
                results.append({**pair, "judge": {"judge_call_error": str(exc)}})
            if i % 100 == 0 or i == len(pairs):
                print(f"  judged {i}/{len(pairs)} (errors: {errors})", file=sys.stderr)

    with open(args.output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(results)} judged pairs to {args.output_path} ({errors} errors)")


if __name__ == "__main__":
    main()
