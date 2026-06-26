"""The contribution standard: a DatasetCard that travels with every contribution.

A card records *what* the data is, *which device* produced it, and — critically —
*consent, license, and anonymization*. The validator (see ``validate.py``) refuses
anything that doesn't carry those, so real-people data can only enter the registry
responsibly. This module is pure data + helpers so a future web portal can reuse it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

SCHEMA_VERSION = "0.1.0"

ALLOWED_CONSENT = {
    "self",                    # the contributor is the only person filmed
    "participants-consented",  # everyone filmed gave informed consent
    "public-domain",           # public-domain / CC0 source
    "public-cc",               # permissively licensed public source
    "synthetic",               # generated, no real people
}
ALLOWED_DEVICES = {"webcam", "phone", "dashcam", "fixed_cam", "glasses", "other"}
ALLOWED_DATA_MODES = {"hosted", "in_repo"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class DatasetCard:
    id: str                      # slug, unique in the registry
    title: str
    contributor: str             # GitHub handle or name
    device: str                  # one of ALLOWED_DEVICES
    task: str                    # what activity was captured
    consent: str                 # one of ALLOWED_CONSENT (never "unspecified")
    license: str                 # e.g. CC-BY-4.0, CC0-1.0
    anonymized: bool             # must be True
    data_mode: str               # hosted | in_repo
    data_url: Optional[str] = None    # hosted: https link to the anonymized dataset
    data_path: Optional[str] = None   # in_repo: relative path to the signal bundle
    num_episodes: int = 0
    num_frames: int = 0
    notes: str = ""
    schema_version: str = SCHEMA_VERSION
    tool_version: str = "0.1.0"
    created_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DatasetCard":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})
