"""
Map the selected QA pairs back to their original rows in the raw ticket CSV.

Writes two outputs:
  - <out>.csv   original ticket format with the matching CSV rows, original columns preserved
  - <out>.json  the same rows paired with their question/answer, for review

Usage:
    python map_to_tickets.py top_250_qa.json Cases_with_state_since_Jan_1_2026.csv top250_tickets
"""

import csv
import json
import re
import sys

CASE_RE = re.compile(r"OSCG\d+")


def open_csv(path):
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            f = open(path, newline="", encoding=encoding)
            f.read()
            f.seek(0)
            return f, encoding
        except UnicodeDecodeError:
            continue
    return open(path, newline="", encoding="utf-8", errors="replace"), "utf-8 (replacement)"


def code_of(text):
    m = CASE_RE.search(text or "")
    return m.group(0) if m else None


def main():
    if len(sys.argv) != 4:
        print("Usage: python map_to_tickets.py <pairs.json> <tickets.csv> <output_prefix>")
        sys.exit(1)
    pairs_path, csv_path, out_prefix = sys.argv[1:4]

    pairs = json.load(open(pairs_path))

    f, encoding = open_csv(csv_path)
    print(f"Reading CSV as {encoding}")
    with f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    by_case = {(r.get("case") or "").strip(): r for r in rows}
    by_code = {}
    for r in rows:
        c = code_of(r.get("case"))
        if c:
            by_code.setdefault(c, r)

    matched_rows = []
    combined = []
    unmatched = []
    matched_by_code = 0

    for p in pairs:
        pid = (p.get("id") or "").strip()
        row = by_case.get(pid)
        if row is None:
            c = code_of(pid)
            row = by_code.get(c) if c else None
            if row is not None:
                matched_by_code += 1
        if row is None:
            unmatched.append(pid)
            continue
        matched_rows.append(row)
        combined.append({
            "case": row.get("case"),
            "short_description": row.get("short_description"),
            "state": row.get("state"),
            "question": p.get("question"),
            "answer": p.get("answer"),
            "selection_tier": p.get("selection_tier"),
            "comments": row.get("comments"),
        })

    csv_out = f"{out_prefix}.csv"
    with open(csv_out, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(matched_rows)

    json_out = f"{out_prefix}.json"
    with open(json_out, "w", encoding="utf-8") as out:
        json.dump(combined, out, ensure_ascii=False, indent=2)

    print(f"Pairs in:            {len(pairs)}")
    print(f"Matched exactly:     {len(matched_rows) - matched_by_code}")
    print(f"Matched by OSCG code:{matched_by_code}")
    print(f"Unmatched:           {len(unmatched)}")
    for u in unmatched[:10]:
        print(f"   {u[:70]}")
    print(f"\nWrote {csv_out} and {json_out}")


if __name__ == "__main__":
    main()
