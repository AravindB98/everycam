"""Parse a 'data-contribution' GitHub issue into an action for the automation.

The website's contribute form posts an issue whose body contains a ```json ... ```
dataset card. This decides what the bot should do:
  - "hosted"   : a valid hosted card -> open a registry PR automatically
  - "guidance" : a signals-only (in_repo) submission -> nudge them to use the CLI
  - "invalid"  : missing/broken/incomplete card -> comment with what to fix

Kept pure (no disk/network) so it is easy to unit-test.
"""

from __future__ import annotations

import json
import re
from typing import Dict

from .validate import validate_card

_JSON_BLOCK = re.compile(r"```json\s*(\{.*?\})\s*```", re.S)


def process_issue(body: str) -> Dict[str, str]:
    """Return {action, id, comment, card_line}."""
    m = _JSON_BLOCK.search(body or "")
    if not m:
        return {
            "action": "invalid", "id": "", "card_line": "",
            "comment": "Thanks for contributing! I couldn't find a dataset-card JSON block in this "
                       "issue. Please use the website's contribute form (it generates one for you).",
        }
    try:
        card = json.loads(m.group(1))
    except Exception as e:  # noqa: BLE001
        return {
            "action": "invalid", "id": "", "card_line": "",
            "comment": f"Thanks! The dataset-card JSON couldn't be parsed:\n\n```\n{e}\n```\n\n"
                       "Please regenerate it from the website form.",
        }

    cid = str(card.get("id", ""))
    # Signals-only data can't travel through an issue — guide them to the CLI.
    if card.get("data_mode") != "hosted":
        return {
            "action": "guidance", "id": cid, "card_line": "",
            "comment": "Thanks! For **signals-only** contributions the data can't come through an "
                       "issue. Please run locally:\n\n```\neverycam contribute --dataset <your_capture> "
                       f"--id {cid or 'my-id'} ... --data-mode in_repo --i-have-rights\n```\n\n"
                       "…then open a pull request — or switch to a **hosted** link and I'll open the "
                       "PR for you automatically.",
        }

    errs = validate_card(card)
    if errs:
        return {
            "action": "invalid", "id": cid, "card_line": "",
            "comment": "Thanks! Before I can add this, please fix:\n\n- " + "\n- ".join(errs)
                       + "\n\nThe website's contribute form fills these in correctly.",
        }
    return {"action": "hosted", "id": cid, "card_line": json.dumps(card) + "\n", "comment": ""}
