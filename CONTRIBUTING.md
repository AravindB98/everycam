# Contributing to EveryCam

Everyone is welcome here — whether you write code or have never opened a terminal. 💛

**New to all this?** Start with the [plain-English explainer](docs/EXPLAINER.md) or the
[project website](https://AravindB98.github.io/everycam/). They explain what EveryCam is, with no
jargon.

There are **two ways to help**, and you don't need to be a programmer for the first one.

---

## 1. Contribute data 🎥 (no coding needed)

This is the most valuable thing most people can do: share a short clip of your hands doing an
everyday task, so robots can learn from it. EveryCam **blurs faces automatically** and can upload
just the *numbers* (no actual video of people).

**The easiest way — the website:** open the
[contribute form](https://AravindB98.github.io/everycam/#contribute), fill in a few boxes, tick the
consent box, and it opens a pre-filled request for you. A maintainer takes it from there.

**Or with one command** (if you've installed the tool):

```bash
bash scripts/webcam_contribute.sh
```

It records ~5 seconds from your webcam, blurs faces, and packages your contribution. Full
step-by-step: [CONTRIBUTING-DATA.md](CONTRIBUTING-DATA.md).

> **Golden rule:** only share videos you have the right to share, where everyone in them is you or
> has said yes. When unsure, don't. EveryCam never tries to identify or track anyone.

---

## 2. Contribute code 💻 (for programmers)

```bash
git clone https://github.com/AravindB98/everycam.git
cd everycam
pip install -e ".[data,dev]"     # core + parquet + test tools
pytest -q                        # all tests should pass (CPU only, no hardware)
ruff check .                     # style check
```

Guidelines:

- **Keep the core lightweight.** New heavy dependencies go behind an optional extra (`[hands]`,
  `[torch]`, …) with a graceful fallback, so `everycam demo` still runs on any laptop.
- **Privacy is non-negotiable.** Changes must keep "anonymize-before-store" and the fail-closed
  behavior. PRs that add face recognition or tracking will be declined.
- **Add a test** for new behavior; run `ruff` before opening a PR.

Good first issues: a new camera adapter, a DNN face/plate backend, the LeRobot video exporter.

---

## What's a "pull request"? (first time on GitHub?)

A **pull request (PR)** is just a friendly "here's my change — want to add it?" On GitHub you make a
copy (a *fork*), commit your change, and click **"Open pull request."** A maintainer reviews it, and
an automated checker (**CI**) runs the tests. Don't worry about getting it perfect — we'll help. 🙌

By contributing, you agree your work is shared under the project's [MIT License](LICENSE).
