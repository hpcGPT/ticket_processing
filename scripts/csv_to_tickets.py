"""
Convert export CSV (columns: case, short_description,
comments, state) into the tickets_combined.json format expected by
HPC-AgentBench's process_tickets.py.

Usage:
    python scripts/csv_to_tickets.py <csv_file_path> hpc_agentbench/tickets_combined.json
"""

import csv
import json
import re
import sys
from collections import defaultdict

HEADER_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} - (.+?) \(Customer Facing Comments\)\s*$",
    re.MULTILINE,
)
SENDER_RE = re.compile(r"^(received from|reply from|submitted by)\s*:\s*(.+)$", re.IGNORECASE)
CODE_TAG_RE = re.compile(r"\[/?code\]", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")

BOILERPLATE_RE = re.compile(
    "|".join([
        r"This case is pending solution acceptance.*",
        r"This case was auto[- ]?closed\.?",
        r"This case is informational only\.?",
        r"It will be auto[- ]?closed if you do not take action\.?",
        r"This case is pending closure.*",
        r"Thank you for your request\. We are currently experiencing a high volume of inquiries.*",
    ]),
    re.IGNORECASE | re.DOTALL,
)

MIN_ENTRY_LENGTH = 20

STAFF_TICKET_THRESHOLD = 5


FORCE_STAFF = set()
FORCE_CUSTOMER = {"Guest User"}


def _clean_body(body: str) -> str:
    body = CODE_TAG_RE.sub("", body)
    body = HTML_TAG_RE.sub("", body)
    match = BOILERPLATE_RE.search(body)
    if match:
        body = body[: match.start()]
    return body.strip()


def extract_entries(raw: str):
    """Return (author, body) tuples for one ticket's comments cell,
    oldest first."""
    if not raw or not raw.strip():
        return []
    matches = list(HEADER_RE.finditer(raw))
    entries = []
    for i, m in enumerate(matches):
        author = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        body = raw[start:end].strip()
        entries.append((author, body))
    entries.reverse()  # oldest first
    return entries


def classify_role(author: str, body: str, staff_authors: set):
    lines = body.splitlines()
    first_line = lines[0].strip() if lines else ""
    if SENDER_RE.match(first_line):
        return "Human", _clean_body("\n".join(lines[1:]))
    if author in FORCE_CUSTOMER:
        return "Human", _clean_body(body)
    if author in FORCE_STAFF or author in staff_authors:
        return "Assistant", _clean_body(body)
    return "Human", _clean_body(body)


def merge_consecutive(turns):
    merged = []
    for role, body in turns:
        # Entries that are nothing but boilerplate collapse to near-empty
        # once stripped; drop them so they can't become the final turn.
        if len(body) < MIN_ENTRY_LENGTH:
            continue
        if merged and merged[-1][0] == role:
            merged[-1] = (role, merged[-1][1] + "\n\n" + body)
        else:
            merged.append((role, body))
    return merged


def _open_csv(csv_path: str):
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            f = open(csv_path, newline="", encoding=encoding)
            f.read()
            f.seek(0)
            return f, encoding
        except UnicodeDecodeError:
            continue
    return open(csv_path, newline="", encoding="utf-8", errors="replace"), "utf-8 (with replacement)"


def csv_to_tickets(csv_path: str, json_path: str, require_assistant_reply: bool = True) -> None:
    f, encoding_used = _open_csv(csv_path)
    print(f"Reading CSV as {encoding_used}")

    with f:
        rows = list(csv.DictReader(f))

    
    author_tickets = defaultdict(set)
    row_entries = {}  # case_id -> (short_desc, [(author, body), ...])
    for row in rows:
        case_id = (row.get("case") or "").strip()
        if not case_id:
            continue
        short_desc = (row.get("short_description") or "").strip()
        entries = extract_entries(row.get("comments") or "")
        row_entries[case_id] = (short_desc, entries)
        for author, _ in entries:
            author_tickets[author].add(case_id)

    staff_authors = {
        author for author, tickets in author_tickets.items()
        if len(tickets) >= STAFF_TICKET_THRESHOLD
    }
    print(f"Detected {len(staff_authors)} likely staff author(s): {sorted(staff_authors)}")

    # classify and build tickets_combined.json 
    tickets = {}
    skipped_empty = 0
    skipped_no_reply = 0

    for case_id, (short_desc, entries) in row_entries.items():
        turns = [classify_role(author, body, staff_authors) for author, body in entries]
        turns = merge_consecutive(turns)

        if not turns:
            skipped_empty += 1
            continue
        if require_assistant_reply and not any(role == "Assistant" for role, _ in turns):
            skipped_no_reply += 1
            continue

        if turns[0][0] == "Human" and short_desc:
            turns[0] = ("Human", f"Subject: {short_desc}\n\n{turns[0][1]}")
        elif turns[0][0] != "Human" and short_desc:
            turns.insert(0, ("Human", f"Subject: {short_desc}"))

        flat = []
        for role, body in turns:
            flat.extend([role, body])

        if case_id in tickets and len(tickets[case_id]) >= len(flat):
            continue
        tickets[case_id] = flat

    with open(json_path, "w", encoding="utf-8") as out:
        json.dump(tickets, out, ensure_ascii=False, indent=2)

    print(f"Wrote {len(tickets)} tickets to {json_path}")
    if skipped_empty:
        print(f"Skipped {skipped_empty} rows with no parseable comment turns")
    if skipped_no_reply:
        print(f"Skipped {skipped_no_reply} rows with no Assistant reply (no usable answer for a QA pair)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python csv_to_tickets.py <input.csv> <tickets_combined.json>")
        sys.exit(1)
    csv_to_tickets(sys.argv[1], sys.argv[2])
