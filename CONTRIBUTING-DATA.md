# Contributing real data from your device

EveryCam gets better the more **real** everyday-camera data the community brings —
your webcam, phone, dashcam, or smart glasses. This guide shows how to contribute
**responsibly**: every contribution is anonymized and carries consent + a license, or
it doesn't get in.

> **The golden rule:** only contribute footage you have the right to share, with
> identities protected. When in doubt, don't.

## What to capture

Short clips of **everyday manipulation and activity** — the raw material for teaching
robots from human video:

- pouring, stacking, opening/closing, wiping, sorting, assembling, cooking steps
- a fixed/overhead view of a workspace while you do a task
- (dashcam) ordinary driving scenes for ego-motion / world-model data

Aim for steady, well-lit clips of the *hands and objects* (or the scene), 10–60s each.

## What you may and may not contribute

**May:** your own footage; footage where **everyone filmed gave informed consent**;
public-domain or permissively-licensed sources.

**May not:** identifiable people who haven't consented; minors; bystanders in private
settings; sensitive locations; anything you don't have the rights to. EveryCam will not
accept data intended to identify or track people.

## How EveryCam protects privacy

1. **Anonymize before storage.** The privacy gate blurs faces and license plates on
   every frame *before* anything is written to disk.
2. **In-repo = signals only.** If you contribute into the repo, it's the derived
   signals (states/actions/affordances/contact) — **never the raw frames**.
3. **Provenance travels with the data** — device, consent, license, and the
   anonymization flag are recorded in the dataset card.

## Step by step

```bash
pip install -e ".[data]"          # core + parquet export

# 1) Capture from your device (faces/plates auto-blurred). Examples:
everycam capture --preset webcam  --out runs/mine/dataset
everycam capture --preset phone   --path my_clip.mp4 --out runs/mine/dataset
everycam capture --preset dashcam --path drive.mp4   --out runs/mine/dataset

# 2) Sanity-check it: stats + a quick model eval
everycam analyze runs/mine/dataset

# 3) Pick how to share the data:
#    (a) hosted  — publish runs/mine/dataset to Hugging Face / Zenodo, copy the URL
#    (b) in_repo — contribute signals only (tiny, no frames)

# 4) Package the contribution (this is the consent gate):
everycam contribute --dataset runs/mine/dataset \
  --id my-kitchen-pours --title "Pouring water, 20 clips" \
  --contributor <your-github-handle> --device webcam \
  --task "pour water into a cup" --consent self --license CC-BY-4.0 \
  --data-mode in_repo --i-have-rights
#   (for hosted, use:  --data-mode hosted --data-url https://huggingface.co/datasets/...)

# 5) Open a PR with the new registry files. CI runs `everycam validate`,
#    a maintainer reviews, and your dataset gets listed.
```

`--i-have-rights` is your explicit attestation that you have the rights/consent to share
the footage. Without it (or without a valid `--consent`/`--license`), the tool refuses.

## Licensing

Pick a license that lets others learn from your data. **CC-BY-4.0** (credit you) is a good
default; **CC0-1.0** maximizes reuse. You keep ownership; you're granting use.

## Review process

1. You open a PR touching `registry/`.
2. CI validates every entry (consent present, license set, `anonymized: true`, schema OK,
   in-repo bundles contain no media).
3. A maintainer reviews for fit and responsible sourcing.
4. Merged → your dataset is listed and counts toward community analyses and benchmarks.

Thank you for helping build an open, **privacy-respecting** foundation for Physical AI.
