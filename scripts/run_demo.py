#!/usr/bin/env python3
"""Convenience entry point for the end-to-end demo.

Equivalent to ``everycam demo``. Runs with zero hardware, network, or GPU:
builds synthetic everyday-camera episodes, anonymizes + perceives + exports a
LeRobot-style dataset, then trains a tiny CPU model and reports metrics.

    python scripts/run_demo.py --episodes 16 --frames 50 --epochs 400
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from everycam.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["demo", *sys.argv[1:]]))
