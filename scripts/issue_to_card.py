#!/usr/bin/env python3
"""Used by the data-contribution GitHub Action.

Reads the issue body from $ISSUE_BODY, decides the action, writes:
  - comment.md   (a friendly reply to post on the issue), and/or
  - card.jsonl   (the registry line to add, for a valid hosted contribution),
and emits `action` / `id` to $GITHUB_OUTPUT.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from everycam.contrib.issue import process_issue  # noqa: E402


def _existing_ids(path: str = "registry/datasets.jsonl") -> set:
    ids = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    try:
                        ids.add(json.loads(line).get("id"))
                    except json.JSONDecodeError:
                        pass
    return ids


def main() -> None:
    r = process_issue(os.environ.get("ISSUE_BODY", ""))
    if r["action"] == "hosted" and r["id"] in _existing_ids():
        r = {
            "action": "invalid", "id": r["id"], "card_line": "",
            "comment": f"Thanks! A contribution with id `{r['id']}` already exists — "
                       "please choose a different id.",
        }
    if r.get("comment"):
        with open("comment.md", "w") as f:
            f.write(r["comment"])
    if r.get("card_line"):
        with open("card.jsonl", "w") as f:
            f.write(r["card_line"])
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write(f"action={r['action']}\nid={r['id']}\n")
    print(f"action={r['action']} id={r['id']}")


if __name__ == "__main__":
    main()
