#!/usr/bin/env bash
# One-command webcam contribution: capture -> anonymize -> analyze -> contribute.
# Runs on YOUR machine (it needs a real camera). macOS may ask you to grant the
# terminal camera access the first time (System Settings > Privacy & Security > Camera).
#
# Usage:  bash scripts/webcam_contribute.sh
set -e
cd "$(dirname "$0")/.."

echo "EveryCam — contribute a clip from your webcam"
read -r -p "Your GitHub handle: " HANDLE
read -r -p "Short id (e.g. ${HANDLE}-pour): " ID
read -r -p "What activity will you do? (e.g. 'pour water into a cup'): " TASK
read -r -p "License [CC-BY-4.0]: " LIC; LIC="${LIC:-CC-BY-4.0}"
FRAMES="${FRAMES:-150}"

echo
echo "Recording ~$((FRAMES/30))s from your webcam — do the activity in view of the camera."
read -r -p "Press Enter to start..."
everycam capture --preset webcam --hands auto --max-frames "$FRAMES" --out "runs/$ID/dataset"

echo "--- analysis (faces/plates were auto-blurred before saving) ---"
everycam analyze "runs/$ID/dataset" --no-train

everycam contribute --dataset "runs/$ID/dataset" --id "$ID" \
  --title "$TASK ($HANDLE)" --contributor "$HANDLE" --device webcam \
  --task "$TASK" --consent self --license "$LIC" --data-mode in_repo --i-have-rights

everycam aggregate
echo
echo "Done. To submit:"
echo "  git checkout -b add-$ID && git add registry && git commit -m 'data: $ID' && git push"
echo "  then open a pull request on GitHub."
