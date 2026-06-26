#!/usr/bin/env bash
# Serve the EveryCam website locally so the in-browser recorder can use your camera.
# (Browsers only allow the camera over https or http://localhost — never file://.)
#
# Usage:  bash scripts/serve_website.sh   then open the printed URL.
cd "$(dirname "$0")/../docs"
PORT="${1:-8000}"
echo "EveryCam is serving locally."
echo "  → open  http://localhost:${PORT}/record.html   to record from your camera"
echo "  → open  http://localhost:${PORT}/             for the website"
echo "Press Ctrl+C to stop."
python3 -m http.server "$PORT"
