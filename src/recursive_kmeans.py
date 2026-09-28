import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans


def choose_k(n_samples, requested_k, min_cluster_size):
    """
    너무 작은 parent cluster를 과도하게 쪼개지 않도록
    실제 사용할 k를 adaptive하게 결정한다.
    """
    if n_samples < 2 * min_cluster_size:
        return 1

    max_reasonable_k = max(
        1,
        n_samples // min_cluster_size
    )

    return min(
        requested_k,
        max_reasonable_k,
        n_samples,
    )


def run_kmeans(
    x,
    requested_k,
    min_cluster_size,
    seed,
):
    actual_k = choose_k(
        len(x),
        requested_k,
        min_cluster_size,
    )

    if actual_k == 1:
        return np.zeros(
            len(x),
            dtype=np.int64,
        )

    model = MiniBatchKMeans(
        n_clusters=actual_k,
        random_state=seed,
        batch_size=min(4096, len(x)),
        n_init=10,
        max_iter=300,
        reassignment_ratio=0.01,
    )

    labels = model.fit_predict(x)

    return labels.astype(np.int64)


def cluster_size_summary(labels):
    _, counts = np.unique(
        labels,
        return_counts=True,
    )

    return {
        "num_clusters": int(len(counts)),
        "min": int(counts.min()),
        "mean": float(counts.mean()),
        "median": float(np.median(counts)),
        "max": int(counts.max()),
    }


def tuple_cluster_summary(label_tuples):
    """
    label_tuples:
        [(global,), ...]
        [(global, local), ...]
        [(global, local, fine), ...]
    """

    counts = {}

    for key in label_tuples:
        key = tuple(key)
        counts[key] = counts.get(key, 0) + 1

    sizes = np.asarray(
        list(counts.values()),
        dtype=np.int64,
    )

    return {
        "num_clusters": int(len(sizes)),
        "min": int(sizes.min()),
        "mean": float(sizes.mean()),
        "median": float(np.median(sizes)),
        "max": int(sizes.max()),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--graph-dir",
        required=True,
    )

    parser.add_argument(
        "--global-k",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--local-k",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--fine-k",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    graph_dir = Path(args.graph_dir)

    # --------------------------------------------------
    # Load
    # --------------------------------------------------

    print("[1/5] Loading embeddings...")

    embeddings = np.load(
        graph_dir / "structural_embeddings.npy"
    ).astype(np.float32)

    lcc_nodes = np.load(
        graph_dir / "lcc_nodes.npy"
    ).astype(np.int64)

    item_ids = np.load(
        graph_dir / "item_ids.npy",
        allow_pickle=True,
    )

    assert len(embeddings) == len(lcc_nodes), (
        f"embedding rows ({len(embeddings)}) "
        f"!= LCC nodes ({len(lcc_nodes)})"
    )

    lcc_item_ids = item_ids[lcc_nodes]

    n_items = len(embeddings)

    print(f"items      : {n_items:,}")
    print(f"embedding  : {embeddings.shape}")

    # --------------------------------------------------
    # Global
    # --------------------------------------------------

    print(
        f"\n[2/5] Global clustering "
        f"(requested K={args.global_k})..."
    )

    global_labels = run_kmeans(
        embeddings,
        requested_k=args.global_k,
        min_cluster_size=args.min_cluster_size,
        seed=args.seed,
    )

    print(
        f"global clusters : "
        f"{len(np.unique(global_labels)):,}"
    )

    # --------------------------------------------------
    # Local
    # --------------------------------------------------

    print(
        f"\n[3/5] Local clustering "
        f"(requested K={args.local_k} per Global)..."
    )

    local_labels = np.full(
        n_items,
        -1,
        dtype=np.int64,
    )

    for global_id in sorted(
        np.unique(global_labels)
    ):
        indices = np.where(
            global_labels == global_id
        )[0]

        child_labels = run_kmeans(
            embeddings[indices],
            requested_k=args.local_k,
            min_cluster_size=args.min_cluster_size,
            seed=args.seed + int(global_id) + 1000,
        )

        local_labels[indices] = child_labels

        print(
            f"  G={global_id:02d} "
            f"items={len(indices):5,d} "
            f"local_clusters="
            f"{len(np.unique(child_labels)):2d}"
        )

    assert np.all(local_labels >= 0)

    # --------------------------------------------------
    # Fine
    # --------------------------------------------------

    print(
        f"\n[4/5] Fine clustering "
        f"(requested K={args.fine_k} per Local)..."
    )

    fine_labels = np.full(
        n_items,
        -1,
        dtype=np.int64,
    )

    local_parent_pairs = sorted(
        set(
            zip(
                global_labels.tolist(),
                local_labels.tolist(),
            )
        )
    )

    for global_id, local_id in local_parent_pairs:

        indices = np.where(
            (global_labels == global_id)
            & (local_labels == local_id)
        )[0]

        child_labels = run_kmeans(
            embeddings[indices],
            requested_k=args.fine_k,
            min_cluster_size=args.min_cluster_size,
            seed=(
                args.seed
                + int(global_id) * 1000
                + int(local_id)
                + 100_000
            ),
        )

        fine_labels[indices] = child_labels

    assert np.all(fine_labels >= 0)

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    print("\n[5/5] Saving hierarchical labels...")

    df = pd.DataFrame(
        {
            "item_id": lcc_item_ids,
            "graph_node": lcc_nodes,
            "global_label": global_labels,
            "local_label": local_labels,
            "fine_label": fine_labels,
        }
    )

    output_csv = (
        graph_dir
        / "hierarchical_labels.csv"
    )

    df.to_csv(
        output_csv,
        index=False,
    )

    # ----------------------------------------------
    # Statistics
    # ----------------------------------------------

    global_paths = [
        (g,)
        for g in global_labels
    ]

    local_paths = list(
        zip(
            global_labels,
            local_labels,
        )
    )

    fine_paths = list(
        zip(
            global_labels,
            local_labels,
            fine_labels,
        )
    )

    stats = {
        "config": {
            "global_k_requested": args.global_k,
            "local_k_requested": args.local_k,
            "fine_k_requested": args.fine_k,
            "min_cluster_size": args.min_cluster_size,
            "seed": args.seed,
        },

        "num_items": int(n_items),

        "global": tuple_cluster_summary(
            global_paths
        ),

        "local": tuple_cluster_summary(
            local_paths
        ),

        "fine": tuple_cluster_summary(
            fine_paths
        ),
    }

    stats_path = (
        graph_dir
        / "cluster_stats.json"
    )

    with open(
        stats_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            stats,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n========== HIERARCHY COMPLETE ==========")

    for level in [
        "global",
        "local",
        "fine",
    ]:
        s = stats[level]

        print(f"\n{level.upper()}")
        print(
            f"clusters : "
            f"{s['num_clusters']:,}"
        )
        print(
            f"size     : "
            f"min={s['min']:,} / "
            f"mean={s['mean']:.2f} / "
            f"median={s['median']:.2f} / "
            f"max={s['max']:,}"
        )

    print("\nSample labels:")

    print(
        df[
            [
                "item_id",
                "global_label",
                "local_label",
                "fine_label",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\nSaved:")
    print(output_csv)
    print(stats_path)


if __name__ == "__main__":
    main()
