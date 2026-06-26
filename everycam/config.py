"""Configuration objects for the EveryCam pipeline.

A ``PipelineConfig`` fully specifies a run: where frames come from, how they
are anonymized, what is perceived, and how the dataset is written. Presets map
the everyday-camera *kinds* (phone, dashcam, fixed/CCTV, glasses) onto sensible
privacy + perception defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Tuple


@dataclass
class SourceConfig:
    kind: str = "synthetic"  # synthetic | webcam | file | stream | image_dir
    path: Optional[str] = None  # file path, stream URL, or directory
    device_index: int = 0  # webcam index
    source_type: str = "synthetic"  # provenance hint (phone|dashcam|fixed_cam|glasses|...)
    device: str = "unknown"
    fps: float = 30.0
    max_frames: int = 150
    resize: Optional[Tuple[int, int]] = None  # (W, H)
    seed: int = 0  # synthetic generator seed


@dataclass
class PrivacyConfig:
    """Anonymization runs BEFORE any frame is stored or analyzed for export."""

    blur_faces: bool = True
    blur_plates: bool = True
    method: str = "gaussian"  # gaussian | pixelate | fill
    strength: int = 35  # blur kernel / pixelation block size
    fail_closed: bool = True  # on detector error, blur conservatively (never leak)
    detect_backend: str = "auto"  # auto | haar | dnn | none


@dataclass
class PerceptionConfig:
    hands: bool = True
    hand_backend: str = "auto"  # auto | mediapipe | heuristic | none
    objects: bool = True
    object_backend: str = "auto"  # auto | heuristic | none
    depth: bool = False
    ego_motion: bool = True


@dataclass
class ExportConfig:
    out_dir: str = "out/everycam_dataset"
    fmt: str = "auto"  # auto | parquet | jsonl  (auto = parquet if pyarrow present)
    save_images: bool = True
    image_format: str = "png"
    task: str = "everyday manipulation demonstration"


@dataclass
class PipelineConfig:
    source: SourceConfig = field(default_factory=SourceConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    consent: str = "unspecified"
    license: str = "unspecified"

    # ----- convenience constructors -------------------------------------------------
    @classmethod
    def demo(cls) -> "PipelineConfig":
        """A fully synthetic run — no hardware, no network, no GPU."""
        return cls(
            source=SourceConfig(
                kind="synthetic", source_type="synthetic", max_frames=60, seed=0
            ),
            consent="synthetic",
            license="CC0-1.0",
        )

    @classmethod
    def from_preset(cls, name: str, path: Optional[str] = None) -> "PipelineConfig":
        """Build a config for a named everyday-camera kind.

        Presets pick provenance + privacy defaults appropriate to the device
        (e.g. dashcams blur license plates hard; fixed/CCTV cams blur faces hard).
        """
        name = name.lower()
        if name not in PRESETS:
            raise ValueError(
                f"Unknown preset '{name}'. Options: {', '.join(sorted(PRESETS))}"
            )
        cfg = PRESETS[name]()
        if path is not None:
            cfg.source.path = path
            if cfg.source.kind == "synthetic":
                cfg.source.kind = "stream" if "://" in path else "file"
        return cfg

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PipelineConfig":
        return cls(
            source=SourceConfig(**d.get("source", {})),
            privacy=PrivacyConfig(**d.get("privacy", {})),
            perception=PerceptionConfig(**d.get("perception", {})),
            export=ExportConfig(**d.get("export", {})),
            consent=d.get("consent", "unspecified"),
            license=d.get("license", "unspecified"),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        import yaml  # optional dep; only needed if you use YAML configs

        with open(path, "r") as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": asdict(self.source),
            "privacy": asdict(self.privacy),
            "perception": asdict(self.perception),
            "export": asdict(self.export),
            "consent": self.consent,
            "license": self.license,
        }


# ----- presets: everyday-camera kinds -> sensible defaults --------------------------
def _webcam() -> PipelineConfig:
    c = PipelineConfig()
    c.source = SourceConfig(kind="webcam", source_type="webcam", device="laptop_webcam")
    c.privacy = PrivacyConfig(blur_faces=True, blur_plates=False)
    c.consent = "self"
    return c


def _phone() -> PipelineConfig:
    c = PipelineConfig()
    c.source = SourceConfig(kind="file", source_type="phone", device="smartphone")
    c.privacy = PrivacyConfig(blur_faces=True, blur_plates=True)
    return c


def _dashcam() -> PipelineConfig:
    c = PipelineConfig()
    c.source = SourceConfig(kind="file", source_type="dashcam", device="dashcam")
    # Driving footage: faces + plates of bystanders must go.
    c.privacy = PrivacyConfig(blur_faces=True, blur_plates=True, strength=45)
    c.perception = PerceptionConfig(hands=False, objects=True, ego_motion=True)
    c.export.task = "egocentric driving scene"
    return c


def _fixed_cam() -> PipelineConfig:
    c = PipelineConfig()
    c.source = SourceConfig(kind="stream", source_type="fixed_cam", device="fixed_camera")
    # Fixed / overhead / CCTV-style: anonymize people hard; identity is never stored.
    c.privacy = PrivacyConfig(blur_faces=True, blur_plates=True, strength=45)
    c.perception = PerceptionConfig(hands=True, objects=True, ego_motion=False)
    c.export.task = "fixed-view activity"
    return c


def _glasses() -> PipelineConfig:
    c = PipelineConfig()
    c.source = SourceConfig(kind="file", source_type="glasses", device="smart_glasses")
    c.privacy = PrivacyConfig(blur_faces=True, blur_plates=True)
    c.export.task = "egocentric manipulation"
    return c


def _phone_ip() -> PipelineConfig:
    # Phone used as a wireless camera (IP Webcam / DroidCam app) over http/rtsp.
    c = PipelineConfig()
    c.source = SourceConfig(kind="stream", source_type="phone", device="phone_ip_camera")
    c.privacy = PrivacyConfig(blur_faces=True, blur_plates=True)
    c.export.task = "phone (wireless) manipulation"
    return c


def _ipcam() -> PipelineConfig:
    # Generic IP / RTSP network camera.
    c = PipelineConfig()
    c.source = SourceConfig(kind="stream", source_type="fixed_cam", device="ip_camera")
    c.privacy = PrivacyConfig(blur_faces=True, blur_plates=True, strength=45)
    c.export.task = "ip-camera activity"
    return c


def _gopro() -> PipelineConfig:
    # Action cam (GoPro/Insta360) recording, or its UVC webcam mode.
    c = PipelineConfig()
    c.source = SourceConfig(kind="file", source_type="glasses", device="action_cam")
    c.privacy = PrivacyConfig(blur_faces=True, blur_plates=True)
    c.export.task = "egocentric action-cam manipulation"
    return c


def _frames() -> PipelineConfig:
    # A folder of already-extracted image frames.
    c = PipelineConfig()
    c.source = SourceConfig(kind="image_dir", source_type="other", device="image_folder")
    c.privacy = PrivacyConfig(blur_faces=True, blur_plates=True)
    c.export.task = "image-folder activity"
    return c


PRESETS = {
    "webcam": _webcam,
    "phone": _phone,
    "phone_ip": _phone_ip,
    "dashcam": _dashcam,
    "fixed_cam": _fixed_cam,
    "cctv": _fixed_cam,   # alias
    "ipcam": _ipcam,
    "rtsp": _ipcam,       # alias
    "glasses": _glasses,
    "gopro": _gopro,
    "frames": _frames,
}
