import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans


def choose_k(n, requested_k, min_cluster_size):
    if n < 2 * min_cluster_size:
        return 1

    return min(
        requested_k,
        max(1, n // min_cluster_size),
        n,
    )


def run_kmeans(x, k, min_cluster_size, seed):
    actual_k = choose_k(
        len(x),
        k,
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
    n = len(embeddings)

    global_labels = run_kmeans(
        embeddings,
        global_k,
        min_cluster_size,
        seed,
    )

    local_labels = np.full(
        n,
        -1,
        dtype=np.int64,
    )

    for g in sorted(np.unique(global_labels)):
        idx = np.where(
            global_labels == g
        )[0]

        local_labels[idx] = run_kmeans(
            embeddings[idx],
            local_k,
            min_cluster_size,
            seed + int(g) + 1000,
        )

    fine_labels = np.full(
        n,
        -1,
        dtype=np.int64,
    )

    parents = sorted(
        set(
            zip(
                global_labels.tolist(),
                local_labels.tolist(),
            )
        )
    )

    for g, l in parents:
        idx = np.where(
            (global_labels == g)
            & (local_labels == l)
        )[0]

        fine_labels[idx] = run_kmeans(
            embeddings[idx],
            fine_k,
            min_cluster_size,
            seed
            + int(g) * 1000
            + int(l)
            + 100_000,
        )

    return (
        global_labels,
        local_labels,
        fine_labels,
    )


def encode_path(*arrays):
    mapping = {}

    labels = np.empty(
        len(arrays[0]),
        dtype=np.int64,
    )

    next_id = 0

    for i, key in enumerate(zip(*arrays)):
        key = tuple(int(x) for x in key)

        if key not in mapping:
            mapping[key] = next_id
            next_id += 1

        labels[i] = mapping[key]

    return labels


def load_lcc_edges(graph_dir):
    edge_data = np.load(
        graph_dir / "edges.npz"
    )

    src = edge_data["src"].astype(np.int64)
    dst = edge_data["dst"].astype(np.int64)
    weight = edge_data["weight"].astype(np.float32)

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


def evaluate(src, dst, weight, labels):
    same = labels[src] == labels[dst]

    intra_edges = int(same.sum())

    intra_ratio = (
        intra_edges / len(src)
        if len(src)
        else 0.0
    )

    intra_weights = weight[same]

    mean_weight = (
        float(intra_weights.mean())
        if len(intra_weights)
        else 0.0
    )

    _, counts = np.unique(
        labels,
        return_counts=True,
    )

    possible_pairs = int(
        sum(
            int(n) * (int(n) - 1) // 2
            for n in counts
        )
    )

    density = (
        intra_edges / possible_pairs
        if possible_pairs
        else 0.0
    )

    return {
        "clusters": int(len(counts)),
        "intra_edge_ratio": float(intra_ratio),
        "intra_mean_weight": float(mean_weight),
        "intra_density": float(density),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--graph-dir",
        required=True,
    )

    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=[1, 2, 3],
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

    src, dst, weight = load_lcc_edges(
        graph_dir
    )

    rows = []

    for layer in args.layers:
        emb_path = (
            graph_dir
            / f"structural_embeddings_L{layer}.npy"
        )

        if not emb_path.exists():
            raise FileNotFoundError(
                f"Missing embedding file: {emb_path}"
            )

        print(
            f"\n========== LIGHTGCN LAYER {layer} =========="
        )

        embeddings = np.load(
            emb_path
        ).astype(np.float32)

        print(
            f"embedding shape : {embeddings.shape}"
        )

        g, l, f = build_hierarchy(
            embeddings,
            args.global_k,
            args.local_k,
            args.fine_k,
            args.min_cluster_size,
            args.seed,
        )

        level_labels = {
            "global": g,
            "local": encode_path(g, l),
            "fine": encode_path(g, l, f),
        }

        for level, labels in level_labels.items():
            result = evaluate(
                src,
                dst,
                weight,
                labels,
            )

            rows.append(
                {
                    "lightgcn_layers": layer,
                    "level": level,
                    **result,
                }
            )

            print(
                f"{level:6s} | "
                f"clusters={result['clusters']:4d} | "
                f"density={result['intra_density']:.6f} | "
                f"mean_weight={result['intra_mean_weight']:.4f} | "
                f"intra_ratio={result['intra_edge_ratio']:.4f}"
            )

    df = pd.DataFrame(rows)

    output = (
        graph_dir
        / "lightgcn_layer_ablation.csv"
    )

    df.to_csv(
        output,
        index=False,
    )

    print(
        "\n========== SUMMARY =========="
    )

    print(
        df[
            [
                "lightgcn_layers",
                "level",
                "clusters",
                "intra_density",
                "intra_mean_weight",
                "intra_edge_ratio",
            ]
        ].to_string(index=False)
    )

    print(
        f"\nSaved: {output}"
    )


if __name__ == "__main__":
    main()
