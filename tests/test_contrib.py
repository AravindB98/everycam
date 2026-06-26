import os

import pytest

from everycam.contrib import (
    DatasetCard,
    analyze_dataset,
    build_contribution,
    register,
    validate_card,
    validate_registry,
)


def _good_card():
    return DatasetCard(
        id="my-test", title="t", contributor="me", device="webcam", task="x",
        consent="self", license="CC-BY-4.0", anonymized=True, data_mode="hosted",
        data_url="https://example.com/d",
    ).to_dict()


def test_good_card_passes():
    assert validate_card(_good_card()) == []


def test_rejects_missing_consent():
    c = _good_card()
    c["consent"] = "unspecified"
    assert any("consent" in e for e in validate_card(c))


def test_rejects_not_anonymized():
    c = _good_card()
    c["anonymized"] = False
    assert any("anonymized" in e for e in validate_card(c))


def test_rejects_unset_license():
    c = _good_card()
    c["license"] = "unspecified"
    assert any("license" in e for e in validate_card(c))


def test_contribute_requires_rights_attestation(demo_dataset, tmp_path):
    with pytest.raises(PermissionError):
        build_contribution(
            demo_dataset, id="x1", title="t", contributor="me", device="webcam",
            task="x", consent="self", license="CC0-1.0", data_mode="in_repo",
            registry_dir=str(tmp_path / "registry"), attest_rights=False,
        )


def test_in_repo_contribution_has_no_images_and_validates(demo_dataset, tmp_path):
    reg = str(tmp_path / "registry")
    card = build_contribution(
        demo_dataset, id="my-data", title="t", contributor="me", device="webcam",
        task="pour", consent="self", license="CC0-1.0", data_mode="in_repo",
        registry_dir=reg, attest_rights=True,
    )
    register(card, registry_dir=reg)
    bundle = os.path.join(reg, card.data_path)
    files = os.listdir(bundle)
    assert "signals.jsonl" in files
    assert not any(f.lower().endswith((".png", ".jpg", ".jpeg", ".mp4", ".mov")) for f in files)
    n, errs = validate_registry(reg)
    assert n == 1 and errs == []


def test_analyze_dataset(demo_dataset):
    s = analyze_dataset(demo_dataset, train=False)
    assert s["total_frames"] > 0
    assert s["anonymized"] is True


def test_issue_hosted_is_pr_ready():
    import json

    from everycam.contrib import process_issue

    card = _good_card()  # hosted + valid
    body = "Hi!\n```json\n" + json.dumps(card) + "\n```\nthanks"
    r = process_issue(body)
    assert r["action"] == "hosted"
    assert r["id"] == "my-test"
    assert card["id"] in r["card_line"]


def test_issue_no_json_is_invalid():
    from everycam.contrib import process_issue

    r = process_issue("just some text, no card here")
    assert r["action"] == "invalid" and r["comment"]


def test_issue_in_repo_gets_guidance():
    import json

    from everycam.contrib import process_issue

    card = _good_card()
    card["data_mode"] = "in_repo"
    card.pop("data_url", None)
    r = process_issue("```json\n" + json.dumps(card) + "\n```")
    assert r["action"] == "guidance"


def test_issue_invalid_card_lists_errors():
    import json

    from everycam.contrib import process_issue

    card = _good_card()
    card["consent"] = "unspecified"
    r = process_issue("```json\n" + json.dumps(card) + "\n```")
    assert r["action"] == "invalid" and "consent" in r["comment"]


def test_aggregate_registry(demo_dataset, tmp_path):
    from everycam.contrib import aggregate_registry

    reg = str(tmp_path / "registry")
    card = build_contribution(
        demo_dataset, id="agg-1", title="t", contributor="me", device="phone",
        task="x", consent="self", license="CC0-1.0", data_mode="in_repo",
        registry_dir=reg, attest_rights=True,
    )
    register(card, registry_dir=reg)
    agg = aggregate_registry(reg)
    assert agg["contributions"] == 1
    assert agg["total_frames"] > 0
    assert agg["by_device"].get("phone") == 1
    assert agg["pooled_contact_ratio"] is not None
