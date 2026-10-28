import hashlib
import json

import pytest

from pipeline.sources import fetch, fetch_selection


def test_cached_source_is_rehashed_and_bound_to_current_view_role(tmp_path):
    data = b"verified sensor bytes"
    path = tmp_path / "view.NTF"
    path.write_bytes(data)
    product = {
        "key": "view.NTF",
        "uri": "https://example.invalid/view.NTF",
        "bytes": len(data),
        "etag": "version",
    }
    evidence = {**product, "sha256": hashlib.sha256(data).hexdigest()}
    path.with_suffix(".json").write_text(json.dumps(evidence))
    result = fetch({**product, "role": "holdout", "catalog_id": "view-1"}, tmp_path)
    assert result["reused"] is True
    assert result["role"] == "holdout" and result["catalog_id"] == "view-1"
    path.write_bytes(b"changed sensor bytes")
    with pytest.raises(ValueError, match="does not match"):
        fetch(product, tmp_path)


def test_source_budget_rejects_before_network_or_directory_creation(tmp_path):
    def product(name):
        return {
            "key": name,
            "uri": f"https://example.invalid/{name}",
            "bytes": 1024**3 + 1,
        }

    selection = {
        "status": "candidate",
        "train": [{"catalog_id": "a", "pan": product("a"), "msi": product("b")}],
        "holdout": [],
    }
    target = tmp_path / "raw"
    with pytest.raises(ValueError, match="budget"):
        fetch_selection(selection, target)
    assert not target.exists()


def test_duplicate_products_cannot_be_counted_as_distinct_views(tmp_path):
    product = {"key": "a", "uri": "https://example.invalid/a", "bytes": 10}
    selection = {
        "status": "candidate",
        "train": [{"catalog_id": "a", "pan": product, "msi": product}],
        "holdout": [],
    }
    with pytest.raises(ValueError, match="duplicate"):
        fetch_selection(selection, tmp_path / "raw")
