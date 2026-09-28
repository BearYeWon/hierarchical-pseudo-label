import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


def load_graph(graph_dir):
    graph_dir = Path(graph_dir)

    edge_data = np.load(graph_dir / "edges.npz")
    item_ids = np.load(
        graph_dir / "item_ids.npy",
        allow_pickle=True,
    )

    src = edge_data["src"].astype(np.int64)
    dst = edge_data["dst"].astype(np.int64)
    weight = edge_data["weight"].astype(np.float32)

    return item_ids, src, dst, weight


def build_adjacency(num_nodes, src, dst, weight):
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    data = np.concatenate([weight, weight])

    adj = sp.csr_matrix(
        (data, (rows, cols)),
        shape=(num_nodes, num_nodes),
        dtype=np.float32,
    )

    adj.eliminate_zeros()
    return adj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-dir", required=True)
    args = parser.parse_args()

    graph_dir = Path(args.graph_dir)

    print("[1/3] Loading graph...")

    item_ids, src, dst, weight = load_graph(graph_dir)

    print(f"nodes : {len(item_ids):,}")
    print(f"edges : {len(src):,}")

    print("\n[2/3] Building adjacency...")

    adj = build_adjacency(
        len(item_ids),
        src,
        dst,
        weight,
    )

    print(f"adjacency shape : {adj.shape}")
    print(f"adjacency nnz   : {adj.nnz:,}")

    print("\n[3/3] Finding connected components...")

    num_components, labels = connected_components(
        adj,
        directed=False,
        return_labels=True,
    )

    component_sizes = np.bincount(labels)

    largest_component_id = int(
        np.argmax(component_sizes)
    )

    lcc_nodes = np.where(
        labels == largest_component_id
    )[0].astype(np.int64)

    isolated_nodes = int(
        np.sum(component_sizes == 1)
    )

    lcc_adj = adj[lcc_nodes][:, lcc_nodes]

    print(f"components           : {num_components:,}")
    print(f"isolated components  : {isolated_nodes:,}")
    print(f"largest component    : {len(lcc_nodes):,}")
    print(f"LCC adjacency shape  : {lcc_adj.shape}")
    print(f"LCC adjacency nnz    : {lcc_adj.nnz:,}")
    print(f"LCC undirected edges : {lcc_adj.nnz // 2:,}")

    np.save(
        graph_dir / "lcc_nodes.npy",
        lcc_nodes,
    )

    stats_path = graph_dir / "graph_stats.json"

    with open(
        stats_path,
        "r",
        encoding="utf-8",
    ) as f:
        stats = json.load(f)

    stats["components"] = int(num_components)
    stats["largest_component"] = int(
        len(lcc_nodes)
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

    print("\nSaved:")
    print(graph_dir / "lcc_nodes.npy")
    print(graph_dir / "graph_stats.json")


if __name__ == "__main__":
    main()
