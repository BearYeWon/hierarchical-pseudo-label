import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def choose_k(n_samples, requested_k, min_cluster_size):
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

    return model.fit_predict(x).astype(np.int64)


def build_hierarchy(
    embeddings,
    global_k,
    local_k,
    fine_k,
    min_cluster_size,
    seed,
):
    n_items = len(embeddings)

    # Global
    global_labels = run_kmeans(
        embeddings,
        requested_k=global_k,
        min_cluster_size=min_cluster_size,
        seed=seed,
    )

    # Local
    local_labels = np.full(
        n_items,
        -1,
        dtype=np.int64,
    )

    for global_id in sorted(
        np.unique(global_labels)
    ):
        idx = np.where(
            global_labels == global_id
        )[0]

        child = run_kmeans(
            embeddings[idx],
            requested_k=local_k,
            min_cluster_size=min_cluster_size,
            seed=seed + int(global_id) + 1000,
        )

        local_labels[idx] = child

    # Fine
    fine_labels = np.full(
        n_items,
        -1,
        dtype=np.int64,
    )

    parent_pairs = sorted(
        set(
            zip(
                global_labels.tolist(),
                local_labels.tolist(),
            )
        )
    )

    for global_id, local_id in parent_pairs:
        idx = np.where(
            (global_labels == global_id)
            & (local_labels == local_id)
        )[0]

        child = run_kmeans(
            embeddings[idx],
            requested_k=fine_k,
            min_cluster_size=min_cluster_size,
            seed=(
                seed
                + int(global_id) * 1000
                + int(local_id)
                + 100_000
            ),
        )

        fine_labels[idx] = child

    return {
        "global": global_labels,
        "local": local_labels,
        "fine": fine_labels,
    }


def tuple_to_ids(*arrays):
    tuples = list(
        zip(
            *[
                arr.tolist()
                for arr in arrays
            ]
        )
    )

    mapping = {}
    encoded = np.empty(
        len(tuples),
        dtype=np.int64,
    )

    next_id = 0

    for i, key in enumerate(tuples):
        key = tuple(
            int(x)
            for x in key
        )

        if key not in mapping:
            mapping[key] = next_id
            next_id += 1

        encoded[i] = mapping[key]

    return encoded


def cluster_size_stats(labels):
    _, counts = np.unique(
        labels,
        return_counts=True,
    )

    return {
        "clusters": int(len(counts)),
        "min": int(counts.min()),
        "mean": float(counts.mean()),
        "median": float(np.median(counts)),
        "max": int(counts.max()),
    }


def evaluate_graph_level(
    src,
    dst,
    weight,
    labels,
):
    same = labels[src] == labels[dst]

    intra_edges = int(
        same.sum()
    )

    intra_edge_ratio = (
        intra_edges / len(src)
        if len(src)
        else 0.0
    )

    intra_weight = weight[same]

    mean_weight = (
        float(intra_weight.mean())
        if len(intra_weight)
        else 0.0
    )

    counts = np.bincount(labels)

    possible_pairs = int(
        sum(
            n * (n - 1) // 2
            for n in counts
        )
    )

    density = (
        intra_edges / possible_pairs
        if possible_pairs > 0
        else 0.0
    )

    return {
        "intra_edges": intra_edges,
        "intra_edge_ratio": float(
            intra_edge_ratio
        ),
        "intra_mean_weight": mean_weight,
        "intra_density": float(
            density
        ),
    }


def load_lcc_edges(graph_dir):
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

    max_idx = int(
        max(
            src.max(),
            dst.max(),
            lcc_nodes.max(),
        )
    )

    old_to_local = np.full(
        max_idx + 1,
        -1,
        dtype=np.int64,
    )

    old_to_local[lcc_nodes] = np.arange(
        len(lcc_nodes)
    )

    src_local = old_to_local[src]
    dst_local = old_to_local[dst]

    mask = (
        (src_local >= 0)
        & (dst_local >= 0)
    )

    return (
        src_local[mask],
        dst_local[mask],
        weight[mask],
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--graph-dir",
        required=True,
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 7, 2026],
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

    args = parser.parse_args()

    graph_dir = Path(
        args.graph_dir
    )

    print("[1/4] Loading embeddings and graph...")

    embeddings = np.load(
        graph_dir
        / "structural_embeddings.npy"
    ).astype(np.float32)

    src, dst, weight = load_lcc_edges(
        graph_dir
    )

    print(
        f"embeddings : {embeddings.shape}"
    )
    print(
        f"LCC edges  : {len(src):,}"
    )

    all_results = {}
    rows = []

    print("\n[2/4] Running hierarchy for each seed...")

    for seed in args.seeds:

        print(
            f"\n========== SEED {seed} =========="
        )

        result = build_hierarchy(
            embeddings,
            global_k=args.global_k,
            local_k=args.local_k,
            fine_k=args.fine_k,
            min_cluster_size=args.min_cluster_size,
            seed=seed,
        )

        global_labels = result["global"]

        local_path = tuple_to_ids(
            result["global"],
            result["local"],
        )

        fine_path = tuple_to_ids(
            result["global"],
            result["local"],
            result["fine"],
        )

        all_results[seed] = {
            "global": global_labels,
            "local": local_path,
            "fine": fine_path,
        }

        for level, labels in [
            ("global", global_labels),
            ("local", local_path),
            ("fine", fine_path),
        ]:

            size_stats = cluster_size_stats(
                labels
            )

            graph_stats = evaluate_graph_level(
                src,
                dst,
                weight,
                labels,
            )

            row = {
                "seed": seed,
                "level": level,
                **size_stats,
                **graph_stats,
            }

            rows.append(row)

            print(
                f"{level:6s} | "
                f"clusters={size_stats['clusters']:4d} | "
                f"density={graph_stats['intra_density']:.6f} | "
                f"mean_weight={graph_stats['intra_mean_weight']:.4f} | "
                f"intra_ratio={graph_stats['intra_edge_ratio']:.4f}"
            )

    print("\n[3/4] Pairwise ARI / NMI...")

    stability_rows = []

    for seed_a, seed_b in combinations(
        args.seeds,
        2,
    ):
        for level in [
            "global",
            "local",
            "fine",
        ]:
            labels_a = all_results[
                seed_a
            ][level]

            labels_b = all_results[
                seed_b
            ][level]

            ari = adjusted_rand_score(
                labels_a,
                labels_b,
            )

            nmi = normalized_mutual_info_score(
                labels_a,
                labels_b,
            )

            stability_rows.append(
                {
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "level": level,
                    "ARI": float(ari),
                    "NMI": float(nmi),
                }
            )

            print(
                f"{seed_a:4d} vs {seed_b:4d} | "
                f"{level:6s} | "
                f"ARI={ari:.4f} | "
                f"NMI={nmi:.4f}"
            )

    print("\n[4/4] Saving results...")

    df_quality = pd.DataFrame(
        rows
    )

    df_stability = pd.DataFrame(
        stability_rows
    )

    quality_path = (
        graph_dir
        / "seed_stability_quality.csv"
    )

    stability_path = (
        graph_dir
        / "seed_stability_ari_nmi.csv"
    )

    df_quality.to_csv(
        quality_path,
        index=False,
    )

    df_stability.to_csv(
        stability_path,
        index=False,
    )

    summary = {
        "seeds": args.seeds,
        "config": {
            "global_k": args.global_k,
            "local_k": args.local_k,
            "fine_k": args.fine_k,
            "min_cluster_size": args.min_cluster_size,
        },
        "quality_mean_by_level": (
            df_quality
            .groupby("level")[
                [
                    "clusters",
                    "intra_edge_ratio",
                    "intra_mean_weight",
                    "intra_density",
                ]
            ]
            .mean()
            .to_dict()
        ),
        "stability_mean_by_level": (
            df_stability
            .groupby("level")[
                [
                    "ARI",
                    "NMI",
                ]
            ]
            .mean()
            .to_dict()
        ),
    }

    json_path = (
        graph_dir
        / "seed_stability_summary.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\nSaved:")
    print(quality_path)
    print(stability_path)
    print(json_path)


if __name__ == "__main__":
    main()
