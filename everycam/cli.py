"""``everycam`` command-line interface.

    everycam demo                         # full synthetic demo, no hardware
    everycam capture --preset webcam      # build a dataset from your laptop cam
    everycam capture --preset dashcam --path drive.mp4
    everycam capture --kind stream --path rtsp://...   # fixed / CCTV-style cam
    everycam train runs/demo/dataset      # train the affordance model
    everycam info  runs/demo/dataset      # show schema + provenance
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from . import __slogan__, __version__


def _save_sample_montage(dataset_dir: str, path: str, n: int = 6) -> Optional[str]:
    try:
        import cv2
        import numpy as np

        from .export import load_episode_arrays

        d = load_episode_arrays(dataset_dir, ep_idx=0, with_images=True)
        imgs = d["images"]
        if imgs is None or len(imgs) == 0:
            return None
        sel = np.linspace(0, len(imgs) - 1, min(n, len(imgs))).astype(int)
        montage = np.hstack([imgs[i] for i in sel])
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        cv2.imwrite(path, montage)
        return path
    except Exception:
        return None


def cmd_demo(args) -> int:
    import shutil

    from .config import PipelineConfig
    from .export import DatasetWriter
    from .models import train_from_dataset
    from .pipeline import Pipeline

    ds = os.path.join(args.out, "dataset")
    run = os.path.join(args.out, "run")
    shutil.rmtree(ds, ignore_errors=True)

    print(f"[everycam] {__slogan__}")
    print(f"[everycam] building {args.episodes} synthetic episodes "
          f"({args.frames} frames each) -> {ds}")
    writer = DatasetWriter(ds, fps=30.0)
    faces = plates = 0
    for seed in range(args.episodes):
        cfg = PipelineConfig.demo()
        cfg.source.seed = seed
        cfg.source.max_frames = args.frames
        pipe = Pipeline(cfg)
        writer.write_episode(pipe.run(keep_images=True), images=pipe.last_images)
        faces += pipe.last_privacy.get("faces", 0)
        plates += pipe.last_privacy.get("plates", 0)
    writer.finalize()
    print(f"[everycam] privacy gate ran on every frame "
          f"(faces blurred={faces}, plates blurred={plates}; identity never stored)")

    print(f"[everycam] training affordance + contact model ({args.epochs} epochs, CPU)")
    _, metrics = train_from_dataset(ds, out_dir=run, epochs=args.epochs)
    _save_sample_montage(ds, os.path.join(run, "sample_frames.png"))
    print(json.dumps(metrics, indent=2))
    print(f"[everycam] grasp error is {metrics['improvement_vs_baseline_pct']}% "
          f"below the predict-the-mean baseline; contact acc "
          f"{metrics['contact_accuracy'] * 100:.0f}%")
    print(f"[everycam] artifacts: {run}  (dataset: {ds})")
    return 0


def cmd_capture(args) -> int:
    from .config import PipelineConfig, SourceConfig
    from .pipeline import Pipeline

    if args.preset:
        cfg = PipelineConfig.from_preset(args.preset, path=args.path)
    else:
        cfg = PipelineConfig()
        cfg.source = SourceConfig(
            kind=args.kind,
            path=args.path,
            device_index=args.device,
            source_type=(args.kind if args.kind != "file" else "file"),
        )
    cfg.source.max_frames = args.max_frames
    if args.out:
        cfg.export.out_dir = args.out

    pipe = Pipeline(cfg)
    episode, out_dir = pipe.run_and_export()
    print(episode.summary())
    print(f"[everycam] privacy: {pipe.last_privacy}")
    print(f"[everycam] dataset written to: {out_dir}")
    return 0


def cmd_train(args) -> int:
    from .models import train_from_dataset

    _, metrics = train_from_dataset(args.dataset, out_dir=args.out, epochs=args.epochs)
    print(json.dumps(metrics, indent=2))
    return 0


def cmd_info(args) -> int:
    from .export import read_info

    info = read_info(args.dataset)
    keys = ("robot_type", "fps", "total_episodes", "total_frames", "total_tasks", "data_format")
    print(json.dumps({k: info.get(k) for k in keys}, indent=2))
    sidecar = os.path.join(args.dataset, "meta", "everycam.json")
    if os.path.exists(sidecar):
        print("--- provenance (meta/everycam.json) ---")
        with open(sidecar) as f:
            print(f.read())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="everycam", description=f"EveryCam — {__slogan__}")
    p.add_argument("--version", action="version", version=f"everycam {__version__}")
    sub = p.add_subparsers(dest="cmd")

    d = sub.add_parser("demo", help="run the full synthetic demo (no hardware/GPU)")
    d.add_argument("--episodes", type=int, default=16)
    d.add_argument("--frames", type=int, default=50)
    d.add_argument("--epochs", type=int, default=400)
    d.add_argument("--out", default="runs/demo")
    d.set_defaults(func=cmd_demo)

    c = sub.add_parser("capture", help="build a dataset from a camera/file/stream/preset")
    c.add_argument("--preset", help="webcam|phone|dashcam|fixed_cam|cctv|glasses")
    c.add_argument("--kind", default="file", help="synthetic|webcam|file|stream|image_dir")
    c.add_argument("--path", help="file path / stream URL / image folder")
    c.add_argument("--device", type=int, default=0, help="webcam device index")
    c.add_argument("--max-frames", dest="max_frames", type=int, default=150)
    c.add_argument("--out", default="runs/capture/dataset")
    c.set_defaults(func=cmd_capture)

    t = sub.add_parser("train", help="train the affordance+contact model on a dataset")
    t.add_argument("dataset")
    t.add_argument("--out", default="runs/train")
    t.add_argument("--epochs", type=int, default=400)
    t.set_defaults(func=cmd_train)

    i = sub.add_parser("info", help="print dataset schema + provenance")
    i.add_argument("dataset")
    i.set_defaults(func=cmd_info)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "cmd", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
