# Dataset card — <title>

Copy this into your hosted dataset's README (Hugging Face, Zenodo, etc.).

- **id:** `<slug>`
- **contributor:** `<github handle / name>`
- **device:** webcam | phone | dashcam | fixed_cam | glasses | other
- **task:** <what activity was captured>
- **episodes / frames:** <n> / <n>
- **consent:** self | participants-consented | public-domain | public-cc
- **license:** CC-BY-4.0 | CC0-1.0 | ...
- **anonymized:** true (faces + plates blurred with EveryCam before storage)

## Description

<2–4 sentences: setting, what the hands/scene are doing, anything notable.>

## Provenance & consent

<Who/what was filmed and under what consent. Confirm you have the rights to share.>

## Format

EveryCam / LeRobot-style dataset: `meta/`, `data/chunk-000/*.parquet`, anonymized
`images/`. See https://github.com/AravindB98/everycam for tooling.
