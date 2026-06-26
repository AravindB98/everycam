"""The contribution gate — strict validation so only responsible data enters.

Used three ways: by ``everycam contribute`` before packaging, by ``everycam
validate`` locally, and by CI on every pull request to the registry.
"""

from __future__ import annotations

import glob
import json
import os
import re
from typing import Any, Dict, List, Tuple

from .card import ALLOWED_CONSENT, ALLOWED_DATA_MODES, ALLOWED_DEVICES

REQUIRED = ["id", "title", "contributor", "device", "task", "consent", "license",
            "anonymized", "data_mode"]
_RAW_MEDIA = ("png", "jpg", "jpeg", "bmp", "mp4", "mov", "avi", "mkv")
_SLUG = re.compile(r"[a-z0-9][a-z0-9_-]{2,48}")


def validate_card(card: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    for k in REQUIRED:
        if card.get(k) in (None, ""):
            errs.append(f"missing required field: {k}")
    cid = card.get("id")
    if cid and not _SLUG.fullmatch(cid):
        errs.append("id must be a slug [a-z0-9_-], 3-49 chars")
    if card.get("device") not in ALLOWED_DEVICES:
        errs.append(f"device must be one of {sorted(ALLOWED_DEVICES)}")
    if card.get("consent") not in ALLOWED_CONSENT:
        errs.append(f"consent must be one of {sorted(ALLOWED_CONSENT)} (not 'unspecified')")
    lic = card.get("license") or ""
    if not lic or lic == "unspecified":
        errs.append("license must be set (e.g. CC-BY-4.0, CC0-1.0)")
    if card.get("anonymized") is not True:
        errs.append("anonymized must be true — run EveryCam's privacy gate before contributing")
    dm = card.get("data_mode")
    if dm not in ALLOWED_DATA_MODES:
        errs.append(f"data_mode must be one of {sorted(ALLOWED_DATA_MODES)}")
    if dm == "hosted" and not str(card.get("data_url") or "").startswith("https://"):
        errs.append("hosted contributions need an https data_url")
    if dm == "in_repo" and not card.get("data_path"):
        errs.append("in_repo contributions need a data_path")
    return errs


def validate_contribution(registry_dir: str, card: Dict[str, Any]) -> List[str]:
    """Card checks plus on-disk checks for in-repo bundles (no raw media allowed)."""
    errs = validate_card(card)
    if card.get("data_mode") == "in_repo" and card.get("data_path"):
        path = os.path.join(registry_dir, card["data_path"])
        if not os.path.isdir(path):
            errs.append(f"in_repo data_path not found: {card['data_path']}")
        else:
            found = []
            for ext in _RAW_MEDIA:
                found += glob.glob(os.path.join(path, "**", f"*.{ext}"), recursive=True)
            if found:
                errs.append(
                    f"in_repo bundle must contain NO raw images/video (found {len(found)}); "
                    "signals only — that keeps real footage out of the repo"
                )
    return errs


def validate_registry(registry_dir: str) -> Tuple[int, List[str]]:
    """Validate every entry in registry/datasets.jsonl. Returns (n_entries, errors)."""
    index = os.path.join(registry_dir, "datasets.jsonl")
    if not os.path.exists(index):
        return 0, [f"registry index not found: {index}"]
    errs: List[str] = []
    seen: set = set()
    n = 0
    with open(index) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            n += 1
            try:
                card = json.loads(line)
            except json.JSONDecodeError as e:
                errs.append(f"line {ln}: invalid JSON ({e})")
                continue
            cid = card.get("id", f"<line {ln}>")
            if cid in seen:
                errs.append(f"line {ln}: duplicate id '{cid}'")
            seen.add(cid)
            for e in validate_contribution(registry_dir, card):
                errs.append(f"[{cid}] {e}")
    return n, errs
