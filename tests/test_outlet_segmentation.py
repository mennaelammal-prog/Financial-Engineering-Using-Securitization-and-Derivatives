import numpy as np
import pandas as pd
import pytest

from trade_marketing_tool.outlet_segmentation import segment_outlets


def _synthetic_outlets():
    """Three well-separated blobs so K-Means recovers exactly 3 clusters."""
    rng = np.random.default_rng(0)
    low = rng.normal(loc=[5, 20], scale=0.5, size=(6, 2))
    mid = rng.normal(loc=[20, 60], scale=0.5, size=(6, 2))
    high = rng.normal(loc=[40, 120], scale=0.5, size=(6, 2))
    data = np.vstack([low, mid, high])
    df = pd.DataFrame(data, columns=["sales_volume", "basket_size"])
    df["outlet"] = [f"P{i}" for i in range(len(df))]
    return df


def test_segment_outlets_produces_requested_cluster_count():
    df = _synthetic_outlets()
    result = segment_outlets(df, features=["sales_volume", "basket_size"], n_clusters=3)
    assert result.assignments["Cluster"].nunique() == 3
    assert len(result.cluster_profile) == 3


def test_segment_labels_rank_by_first_feature():
    df = _synthetic_outlets()
    result = segment_outlets(
        df, features=["sales_volume", "basket_size"], n_clusters=3, id_column="outlet"
    )
    ordered = result.cluster_profile.sort_values("sales_volume")
    assert list(ordered["Segment"]) == ["Low-Value", "Mid-Value", "High-Value"]


def test_segment_outlets_requires_enough_rows():
    df = _synthetic_outlets().head(2)
    with pytest.raises(ValueError):
        segment_outlets(df, features=["sales_volume", "basket_size"], n_clusters=3)


def test_segment_outlets_rejects_missing_feature():
    df = _synthetic_outlets()
    with pytest.raises(ValueError):
        segment_outlets(df, features=["not_a_column"], n_clusters=3)
