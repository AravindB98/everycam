"""Package a captured dataset into a registry contribution (card + optional bundle).

In-repo bundles deliberately carry **signals only** (states/actions/affordances/
contact) and never raw frames, so real footage stays off the repo. Hosted
contributions keep the full anonymized dataset wherever the contributor published it.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from ..export import load_dataset_arrays, read_info
from .card import DatasetCard
from .validate import validate_contribution


def _read_provenance(dataset_dir: str) -> dict:
    p = os.path.join(dataset_dir, "meta", "everycam.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def make_signal_bundle(dataset_dir: str, out_dir: str) -> int:
    """Write signals only (no images) for an in-repo contribution. Returns #rows."""
    d = load_dataset_arrays(dataset_dir, with_images=False)
    os.makedirs(out_dir, exist_ok=True)
    states, actions, affs, cons = d["states"], d["actions"], d["affordances"], d["contacts"]
    with open(os.path.join(out_dir, "signals.jsonl"), "w") as f:
        for i in range(len(states)):
            f.write(json.dumps({
                "observation.state": [float(x) for x in states[i]],
                "action": [float(x) for x in actions[i]],
                "observation.affordance_xy": [float(x) for x in affs[i]],
                "contact": bool(cons[i]),
            }) + "\n")
    prov = _read_provenance(dataset_dir)
    with open(os.path.join(out_dir, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)
    return len(states)


def build_contribution(
    dataset_dir: str,
    *,
    id: str,
    title: str,
    contributor: str,
    device: str,
    task: str,
    consent: str,
    license: str,
    data_mode: str,
    data_url: Optional[str] = None,
    registry_dir: str = "registry",
    attest_rights: bool = False,
    notes: str = "",
) -> DatasetCard:
    if not attest_rights:
        raise PermissionError(
            "Refusing: you must attest you have the rights/consent to share this footage "
            "(pass attest_rights=True / --i-have-rights)."
        )
    info = {}
    if os.path.exists(os.path.join(dataset_dir, "meta", "info.json")):
        info = read_info(dataset_dir)
    everycam_meta = _read_provenance(dataset_dir)
    prov = everycam_meta.get("provenance", {})
    anonymized = bool(everycam_meta.get("privacy", {}).get("anonymized")
                      or prov.get("anonymized", False))

    card = DatasetCard(
        id=id, title=title, contributor=contributor, device=device, task=task,
        consent=consent, license=license, anonymized=anonymized, data_mode=data_mode,
        data_url=data_url, num_episodes=int(info.get("total_episodes", 0)),
        num_frames=int(info.get("total_frames", 0)), notes=notes,
    )
    if data_mode == "in_repo":
        rel = os.path.join("data", id)
        n = make_signal_bundle(dataset_dir, os.path.join(registry_dir, rel))
        card.data_path = rel
        if not card.num_frames:
            card.num_frames = n

    errs = validate_contribution(registry_dir, card.to_dict())
    if errs:
        raise ValueError("Contribution failed validation:\n  - " + "\n  - ".join(errs))
    return card


def register(card: DatasetCard, registry_dir: str = "registry") -> str:
    """Append a validated card to registry/datasets.jsonl (id must be unique)."""
    os.makedirs(registry_dir, exist_ok=True)
    index = os.path.join(registry_dir, "datasets.jsonl")
    existing = set()
    if os.path.exists(index):
        with open(index) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    try:
                        existing.add(json.loads(line).get("id"))
                    except json.JSONDecodeError:
                        pass
    if card.id in existing:
        raise ValueError(f"id '{card.id}' already in the registry; choose another.")
    with open(index, "a") as f:
        f.write(json.dumps(card.to_dict()) + "\n")
    return index
