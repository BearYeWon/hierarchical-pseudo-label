import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def safe_mean(x):
    if len(x) == 0:
        return 0.0
    return float(np.mean(x))


def cluster_pair_capacity(keys):
    """
    Sum over clusters:
        n_c * (n_c - 1) / 2

    = number of possible item pairs that lie inside clusters.
    """
    counts = {}

    for key in keys:
        key = tuple(key)
        counts[key] = counts.get(key, 0) + 1

    capacity = sum(
        n * (n - 1) // 2
        for n in counts.values()
    )

    return int(capacity)


def evaluate_level(
    name,
    src_local,
    dst_local,
    weights,
    keys,
):
    """
    keys[i] = hierarchical cluster key for LCC-local node i.

    Examples:
      Global : (g,)
      Local  : (g, l)
      Fine   : (g, l, f)
    """

    same_cluster = np.array(
        [
            keys[u] == keys[v]
            for u, v in zip(src_local, dst_local)
        ],
        dtype=bool,
    )

    intra_weights = weights[same_cluster]

    intra_edges = int(
        same_cluster.sum()
    )

    total_edges = len(weights)

    intra_edge_ratio = (
        intra_edges / total_edges
        if total_edges > 0
        else 0.0
    )

    pair_capacity = cluster_pair_capacity(
        keys
    )

    # How densely possible within-cluster pairs
    # are actually connected.
    intra_density = (
        intra_edges / pair_capacity
        if pair_capacity > 0
        else 0.0
    )

    unique_clusters = len(
        set(keys)
    )

    return {
        "level": name,
        "num_clusters": int(unique_clusters),
        "intra_edges": intra_edges,
        "total_edges": int(total_edges),
        "intra_edge_ratio": float(
            intra_edge_ratio
        ),
        "intra_mean_weight": safe_mean(
            intra_weights
        ),
        "intra_median_weight": (
            float(np.median(intra_weights))
            if len(intra_weights)
            else 0.0
        ),
        "possible_intra_pairs": int(
            pair_capacity
        ),
        "intra_density": float(
            intra_density
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--graph-dir",
        required=True,
    )

    args = parser.parse_args()

    graph_dir = Path(
        args.graph_dir
    )

    print("[1/4] Loading graph and hierarchy...")

    edge_data = np.load(
        graph_dir / "edges.npz"
    )

    src = edge_data["src"].astype(
        np.int64
    )
    dst = edge_data["dst"].astype(
        np.int64
    )
    weight = edge_data["weight"].astype(
        np.float32
    )

    lcc_nodes = np.load(
        graph_dir / "lcc_nodes.npy"
    ).astype(np.int64)

    labels_df = pd.read_csv(
        graph_dir / "hierarchical_labels.csv",
        dtype={"item_id": str},
    )

    print(
        f"full graph edges : {len(src):,}"
    )
    print(
        f"LCC nodes        : {len(lcc_nodes):,}"
    )
    print(
        f"label rows       : {len(labels_df):,}"
    )

    # --------------------------------------------------
    # Original graph index -> LCC-local index
    # --------------------------------------------------

    print("\n[2/4] Mapping LCC edges...")

    max_node = int(
        max(
            src.max(),
            dst.max(),
            lcc_nodes.max(),
        )
    )

    old_to_local = np.full(
        max_node + 1,
        -1,
        dtype=np.int64,
    )

    old_to_local[lcc_nodes] = np.arange(
        len(lcc_nodes)
    )

    src_local = old_to_local[src]
    dst_local = old_to_local[dst]

    lcc_edge_mask = (
        (src_local >= 0)
        & (dst_local >= 0)
    )

    src_local = src_local[
        lcc_edge_mask
    ]
    dst_local = dst_local[
        lcc_edge_mask
    ]
    lcc_weights = weight[
        lcc_edge_mask
    ]

    print(
        f"LCC edges        : "
        f"{len(src_local):,}"
    )

    # --------------------------------------------------
    # Verify row ordering
    # --------------------------------------------------

    expected_graph_nodes = (
        labels_df["graph_node"]
        .to_numpy(dtype=np.int64)
    )

    if not np.array_equal(
        expected_graph_nodes,
        lcc_nodes,
    ):
        raise RuntimeError(
            "hierarchical_labels.csv ordering "
            "does not match lcc_nodes.npy"
        )

    # --------------------------------------------------
    # Hierarchical keys
    # --------------------------------------------------

    global_labels = (
        labels_df["global_label"]
        .to_numpy(dtype=np.int64)
    )

    local_labels = (
        labels_df["local_label"]
        .to_numpy(dtype=np.int64)
    )

    fine_labels = (
        labels_df["fine_label"]
        .to_numpy(dtype=np.int64)
    )

    global_keys = [
        (int(g),)
        for g in global_labels
    ]

    local_keys = [
        (int(g), int(l))
        for g, l in zip(
            global_labels,
            local_labels,
        )
    ]

    fine_keys = [
        (int(g), int(l), int(f))
        for g, l, f in zip(
            global_labels,
            local_labels,
            fine_labels,
        )
    ]

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    print("\n[3/4] Evaluating hierarchy...")

    global_stats = evaluate_level(
        "global",
        src_local,
        dst_local,
        lcc_weights,
        global_keys,
    )

    local_stats = evaluate_level(
        "local",
        src_local,
        dst_local,
        lcc_weights,
        local_keys,
    )

    fine_stats = evaluate_level(
        "fine",
        src_local,
        dst_local,
        lcc_weights,
        fine_keys,
    )

    stats = {
        "num_lcc_nodes": int(
            len(lcc_nodes)
        ),
        "num_lcc_edges": int(
            len(src_local)
        ),
        "global": global_stats,
        "local": local_stats,
        "fine": fine_stats,
    }

    # Parent edge retention:
    # Of edges surviving one hierarchy depth,
    # how many survive the next depth?
    stats["local"]["parent_edge_retention"] = (
        local_stats["intra_edges"]
        / global_stats["intra_edges"]
        if global_stats["intra_edges"] > 0
        else 0.0
    )

    stats["fine"]["parent_edge_retention"] = (
        fine_stats["intra_edges"]
        / local_stats["intra_edges"]
        if local_stats["intra_edges"] > 0
        else 0.0
    )

    output_path = (
        graph_dir
        / "hierarchy_evaluation.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            stats,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------
    # Display
    # --------------------------------------------------

    print("\n[4/4] Results")

    print(
        "\n========== HIERARCHY EVALUATION =========="
    )

    for level in [
        "global",
        "local",
        "fine",
    ]:
        s = stats[level]

        print(
            f"\n{level.upper()}"
        )

        print(
            f"clusters              : "
            f"{s['num_clusters']:,}"
        )

        print(
            f"intra edges           : "
            f"{s['intra_edges']:,} / "
            f"{s['total_edges']:,}"
        )

        print(
            f"intra edge ratio      : "
            f"{s['intra_edge_ratio']:.4f}"
        )

        print(
            f"intra mean weight     : "
            f"{s['intra_mean_weight']:.4f}"
        )

        print(
            f"intra median weight   : "
            f"{s['intra_median_weight']:.4f}"
        )

        print(
            f"intra density         : "
            f"{s['intra_density']:.6f}"
        )

        if "parent_edge_retention" in s:
            print(
                f"parent edge retention : "
                f"{s['parent_edge_retention']:.4f}"
            )

    print("\n---------- Expected pattern ----------")
    print(
        "Intra edge ratio: "
        "Global >= Local >= Fine "
        "(expected by nested partition)"
    )

    print(
        "Mean weight / density: "
        "ideally Global < Local < Fine"
    )

    print(
        f"\nSaved: {output_path}"
    )


if __name__ == "__main__":
    main()
