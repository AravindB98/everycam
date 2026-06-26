# EveryCam community data registry

This folder indexes datasets contributed from **real devices**. Each line in
[`datasets.jsonl`](datasets.jsonl) is a `DatasetCard` validated against
[`schema.json`](schema.json). Two kinds of contribution are supported:

- **hosted** — you publish your anonymized dataset somewhere durable (Hugging Face,
  Zenodo, etc.) and register a card with an `https` `data_url`.
- **in_repo** — a tiny **signals-only** bundle (states / actions / affordances / contact —
  **never raw frames**) committed under `registry/data/<id>/`.

## Contribute

Full guide: [CONTRIBUTING-DATA.md](../CONTRIBUTING-DATA.md). In short:

```bash
everycam capture --preset webcam --out runs/mine/dataset      # anonymized capture
everycam analyze runs/mine/dataset                            # sanity stats
everycam contribute --dataset runs/mine/dataset --id my-id \
  --title "..." --contributor <you> --device webcam --task "..." \
  --consent self --license CC-BY-4.0 --data-mode in_repo --i-have-rights
# then open a PR — CI runs `everycam validate`, a maintainer reviews, it gets listed.
```

## The rules (enforced, not optional)

Every entry must carry **consent**, a **license**, and **`anonymized: true`**, or the
validator rejects it. In-repo bundles must contain **no images or video**. EveryCam has
no capability to identify or track people, and contributions that would require it are
not accepted. See [PRIVACY.md](../PRIVACY.md).
