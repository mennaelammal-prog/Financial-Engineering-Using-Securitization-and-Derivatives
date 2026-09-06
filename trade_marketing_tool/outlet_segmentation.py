"""
Outlet/distributor segmentation: factor analysis (PCA) + K-Means clustering.

The trade-marketing application of the cluster/factor-analysis chapter in
*Analytical Statistics with SPSS Applications*: reduce several correlated
outlet metrics (sales volume, basket size, footfall, ...) to their
principal factors, then group outlets into segments so the most profitable
tier can be targeted with a distinct trade strategy.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class SegmentationResult:
    assignments: pd.DataFrame          # index + features + Cluster + Segment label
    cluster_profile: pd.DataFrame      # per-cluster mean of each feature + N, ranked
    n_clusters: int
    features: list[str]
    explained_variance_ratio: list[float]  # from the 2-component factor/PCA projection
    inertia: float                      # K-Means within-cluster sum of squares


_SEGMENT_NAMES = ["Low-Value", "Mid-Value", "High-Value", "Premium", "Elite"]


def segment_outlets(
    data: pd.DataFrame,
    features: list[str],
    n_clusters: int = 3,
    id_column: str | None = None,
    random_state: int = 42,
) -> SegmentationResult:
    """
    Standardize ``features``, project them onto 2 principal factors (for
    visualization / factor-loading interpretation), and cluster outlets
    into ``n_clusters`` segments with K-Means. Segments are labeled by
    ranking cluster means on the first feature (typically sales volume),
    so "High-Value" is always the most profitable-looking cluster.
    """
    if len(features) < 1:
        raise ValueError("need at least 1 feature to segment on")
    missing = [f for f in features if f not in data.columns]
    if missing:
        raise ValueError(f"features not found in data: {missing}")
    n = len(data)
    if n < n_clusters:
        raise ValueError(f"need at least {n_clusters} outlets to form {n_clusters} clusters")

    X = data[features].astype(float).to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_components = min(2, len(features))
    pca = PCA(n_components=n_components, random_state=random_state)
    factors = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    assignments = data.copy()
    assignments["Cluster"] = labels
    for i in range(n_components):
        assignments[f"Factor{i + 1}"] = factors[:, i]

    cluster_profile = (
        assignments.groupby("Cluster")[features]
        .mean()
        .assign(N=assignments.groupby("Cluster").size())
        .reset_index()
        .sort_values(features[0])
        .reset_index(drop=True)
    )
    rank_to_name = {
        cluster_profile.loc[i, "Cluster"]: _SEGMENT_NAMES[
            min(i, len(_SEGMENT_NAMES) - 1)
        ]
        for i in range(len(cluster_profile))
    }
    cluster_profile["Segment"] = cluster_profile["Cluster"].map(rank_to_name)
    assignments["Segment"] = assignments["Cluster"].map(rank_to_name)

    if id_column and id_column in data.columns:
        cols = [id_column] + features + ["Cluster", "Segment"] + [
            f"Factor{i + 1}" for i in range(n_components)
        ]
        assignments = assignments[cols]

    return SegmentationResult(
        assignments=assignments,
        cluster_profile=cluster_profile,
        n_clusters=n_clusters,
        features=features,
        explained_variance_ratio=[float(v) for v in pca.explained_variance_ratio_],
        inertia=float(kmeans.inertia_),
    )


def format_report(result: SegmentationResult) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("        OUTLET SEGMENTATION — K-MEANS + PCA")
    lines.append("=" * 60)
    lines.append(f"Features: {', '.join(result.features)}")
    lines.append(f"Clusters: {result.n_clusters}   Inertia: {result.inertia:.2f}")
    lines.append(
        "Variance explained by factors: "
        + ", ".join(f"{v:.1%}" for v in result.explained_variance_ratio)
    )
    lines.append("")
    lines.append(result.cluster_profile.to_string(index=False))
    return "\n".join(lines)
