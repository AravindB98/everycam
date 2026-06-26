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
    if getattr(args, "hands", None):
        cfg.perception.hand_backend = args.hands  # auto|mediapipe|heuristic|none
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


def cmd_benchmark(args) -> int:
    from .benchmark import run_benchmark

    print("[everycam] running cross-device generalization benchmark "
          f"({args.episodes} episodes/device, {args.epochs} epochs, CPU)")
    s = run_benchmark(episodes=args.episodes, frames=args.frames, epochs=args.epochs, out_dir=args.out)
    print(json.dumps({k: s[k] for k in (
        "devices", "in_device_grasp_mae", "cross_device_grasp_mae", "generalization_gap")}, indent=2))
    ratio = s["cross_device_grasp_mae"] / max(s["in_device_grasp_mae"], 1e-9)
    print(f"[everycam] cross-device grasp error is {ratio:.1f}x the in-device error "
          f"— the 'brittle grounding' gap. Matrix + heatmap in {args.out}")
    return 0


def cmd_backends(args) -> int:
    def _ok(mod):
        try:
            __import__(mod)
            return True
        except Exception:
            return False

    rows = {
        "opencv (core)": _ok("cv2"),
        "mediapipe (3D hand tracking)": _ok("mediapipe"),
        "pandas + pyarrow (parquet export)": _ok("pandas") and _ok("pyarrow"),
        "torch (neural model head)": _ok("torch"),
    }
    print("EveryCam optional backends:")
    for name, ok in rows.items():
        print(f"  [{'x' if ok else ' '}] {name}")
    print("\nThe core runs on numpy + opencv alone; the rest are optional upgrades.")
    return 0


def cmd_contribute(args) -> int:
    from .contrib import build_contribution, register

    try:
        card = build_contribution(
            args.dataset, id=args.id, title=args.title, contributor=args.contributor,
            device=args.device, task=args.task, consent=args.consent, license=args.license,
            data_mode=args.data_mode, data_url=args.data_url, registry_dir=args.registry,
            attest_rights=args.i_have_rights, notes=args.notes or "",
        )
    except (PermissionError, ValueError) as e:
        print(f"[everycam] contribution rejected:\n{e}")
        return 2
    register(card, registry_dir=args.registry)
    print(f"[everycam] '{card.id}' added to {args.registry}/datasets.jsonl "
          f"({card.num_frames} frames, device={card.device}, consent={card.consent})")
    if card.data_mode == "in_repo":
        print(f"[everycam] signal bundle (no images): {args.registry}/{card.data_path}/")
    print("[everycam] next: open a PR ->")
    print(f"  git checkout -b add-{card.id} && git add {args.registry} && "
          f"git commit -m 'data: add {card.id}' && git push")
    print("  CI validates it, a maintainer reviews, then it's listed in the registry.")
    return 0


def cmd_validate(args) -> int:
    from .contrib import validate_card, validate_registry

    if args.card:
        with open(args.card) as f:
            errs = validate_card(json.load(f))
        n = 1
    else:
        n, errs = validate_registry(args.registry)
    if errs:
        print(f"[everycam] {len(errs)} problem(s) across {n} entr(ies):")
        for e in errs:
            print("  -", e)
        return 1
    print(f"[everycam] OK — {n} registry entr{'y' if n == 1 else 'ies'} valid.")
    return 0


def cmd_analyze(args) -> int:
    from .contrib import analyze_dataset

    print(json.dumps(analyze_dataset(args.dataset, out_dir=args.out, train=not args.no_train), indent=2))
    return 0


def cmd_aggregate(args) -> int:
    from .contrib import aggregate_registry, write_report_md

    agg = aggregate_registry(args.registry)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "report.json"), "w") as f:
            json.dump(agg, f, indent=2)
    report_path = args.report or os.path.join(args.registry, "REPORT.md")
    write_report_md(agg, report_path)
    print(json.dumps({k: agg[k] for k in (
        "contributions", "total_frames", "by_device", "by_consent", "pooled_contact_ratio")}, indent=2))
    print(f"[everycam] community report -> {report_path}")
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
    c.add_argument("--hands", choices=["auto", "mediapipe", "heuristic", "none"],
                   default="auto", help="hand backend (auto picks MediaPipe if installed)")
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

    b = sub.add_parser("benchmark", help="cross-device generalization benchmark (no hardware)")
    b.add_argument("--episodes", type=int, default=8)
    b.add_argument("--frames", type=int, default=40)
    b.add_argument("--epochs", type=int, default=250)
    b.add_argument("--out", default="runs/benchmark")
    b.set_defaults(func=cmd_benchmark)

    bk = sub.add_parser("backends", help="show which optional backends are installed")
    bk.set_defaults(func=cmd_backends)

    co = sub.add_parser("contribute", help="package a captured dataset into a registry contribution")
    co.add_argument("--dataset", required=True, help="exported EveryCam dataset dir (from `everycam capture`)")
    co.add_argument("--id", required=True, help="unique slug, e.g. my-kitchen-pours")
    co.add_argument("--title", required=True)
    co.add_argument("--contributor", required=True, help="your GitHub handle or name")
    co.add_argument("--device", required=True,
                    choices=["webcam", "phone", "dashcam", "fixed_cam", "glasses", "other"])
    co.add_argument("--task", required=True, help="what activity was captured")
    co.add_argument("--consent", required=True,
                    help="self | participants-consented | public-domain | public-cc")
    co.add_argument("--license", required=True, help="e.g. CC-BY-4.0, CC0-1.0")
    co.add_argument("--data-mode", dest="data_mode", required=True, choices=["hosted", "in_repo"])
    co.add_argument("--data-url", dest="data_url", help="hosted: https link to your anonymized dataset")
    co.add_argument("--i-have-rights", dest="i_have_rights", action="store_true",
                    help="attest you have the rights/consent to share this footage (required)")
    co.add_argument("--registry", default="registry")
    co.add_argument("--notes", default="")
    co.set_defaults(func=cmd_contribute)

    va = sub.add_parser("validate", help="validate the contribution registry (or one card)")
    va.add_argument("--registry", default="registry")
    va.add_argument("--card", help="validate a single card JSON file instead of the registry")
    va.set_defaults(func=cmd_validate)

    an = sub.add_parser("analyze", help="stats + model eval on any dataset (real or synthetic)")
    an.add_argument("dataset")
    an.add_argument("--out", default=None)
    an.add_argument("--no-train", dest="no_train", action="store_true", help="skip model eval")
    an.set_defaults(func=cmd_analyze)

    ag = sub.add_parser("aggregate", help="pool all registered datasets into a community report")
    ag.add_argument("--registry", default="registry")
    ag.add_argument("--out", default=None, help="dir for report.json")
    ag.add_argument("--report", default=None, help="path for REPORT.md (default registry/REPORT.md)")
    ag.set_defaults(func=cmd_aggregate)
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
