# Contributing to EveryCam

Thanks for your interest! EveryCam aims to be the friendly, responsible capture layer for Physical AI, and
contributions of all sizes are welcome.

## Development setup

```bash
git clone https://github.com/AravindB98/everycam.git
cd everycam
pip install -e ".[data,dev]"     # core + parquet + pytest/ruff
pytest -q                        # should be all green (CPU only, no hardware)
ruff check .                     # lint
```

The whole test suite runs on a bare CPU with synthetic data — no camera, GPU, or network needed.

## Guidelines

- **Keep the core dependency-light.** New heavy dependencies (deep models, etc.) belong behind an optional extra
  (`[hands]`, `[torch]`, ...) with a graceful fallback so `everycam demo` still runs on any laptop.
- **Privacy is non-negotiable.** Changes must preserve anonymize-before-store and the fail-closed behavior. See
  [PRIVACY.md](PRIVACY.md). PRs adding identification/tracking will be declined.
- **Add a test** for new behavior, and run `ruff` before opening a PR.
- Keep functions small and documented; match the existing style.

## Good first issues

- A new source adapter (e.g. a specific IP-camera protocol).
- A DNN face/plate backend for the privacy gate (behind an extra).
- Native LeRobot `mp4` video export + a one-line `LeRobotDataset` loader.
- A monocular-depth backend to produce 3D affordances.

## Pull requests

1. Fork and branch from `main`.
2. Make focused changes with tests.
3. Ensure `pytest` and `ruff check` pass.
4. Open a PR describing the change and its motivation.

By contributing you agree your work is licensed under the project's [MIT License](LICENSE).
