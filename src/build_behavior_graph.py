import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def first_nonempty(value):
    if isinstance(value, list):
        for x in value:
            if isinstance(x, str) and x.strip():
                return x.strip()
        return None

    if isinstance(value, str) and value.strip():
        return value.strip()

    return None


def get_image_url(meta):
    for key in ["imageURLHighRes", "imageURL", "image"]:
        url = first_nonempty(meta.get(key))
        if url:
            return url
    return None


def load_valid_metadata(meta_path):
    """
    Prototype filtering:
    keep items having both title and image.
    """

    metadata = {}

    print("[1/6] Loading metadata...")

    with gzip.open(meta_path, "rt", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            obj = json.loads(line)

            asin = obj.get("asin")
            if not asin:
                continue

            asin = str(asin)
            title = str(obj.get("title") or "").strip()
            image = get_image_url(obj)

            if title and image and asin not in metadata:
                metadata[asin] = {
                    "title": title,
                    "image": image,
                }

            if idx % 100_000 == 0:
                print(
                    f"  metadata lines={idx:,}, "
                    f"valid items={len(metadata):,}"
                )

    print(f"  valid multimodal items={len(metadata):,}")

    return metadata


def load_sequences(
    reviews_path,
    valid_items,
    min_user_interactions=5,
):
    """
    1. Keep valid multimodal items.
    2. Remove exact duplicate events.
    3. Keep users with >= min_user_interactions.
    4. Sort chronologically.
    5. Leave last two events out of train.
    """

    print("\n[2/6] Loading interaction events...")

    seen = set()
    events = []

    with gzip.open(reviews_path, "rt", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            obj = json.loads(line)

            user = obj.get("reviewerID")
            asin = obj.get("asin")
            timestamp = obj.get("unixReviewTime")

            if user is None or asin is None or timestamp is None:
                continue

            user = str(user)
            asin = str(asin)
            timestamp = int(timestamp)

            if asin not in valid_items:
                continue

            key = (user, asin, timestamp)

            if key in seen:
                continue

            seen.add(key)
            events.append(key)

            if idx % 250_000 == 0:
                print(
                    f"  reviews={idx:,}, "
                    f"kept events={len(events):,}"
                )

    print(f"  deduplicated multimodal events={len(events):,}")

    user_counts = Counter(user for user, _, _ in events)

    valid_users = {
        user
        for user, count in user_counts.items()
        if count >= min_user_interactions
    }

    print(
        f"  users with >= {min_user_interactions} interactions="
        f"{len(valid_users):,}"
    )

    sequences = defaultdict(list)

    for user, asin, timestamp in events:
        if user in valid_users:
            sequences[user].append((timestamp, asin))

    train_sequences = {}

    for user, seq in sequences.items():
        # stable chronological ordering
        seq.sort(key=lambda x: (x[0], x[1]))

        items = [asin for _, asin in seq]

        # validation = second-last, test = last
        train_items = items[:-2]

        if train_items:
            train_sequences[user] = train_items

    train_interactions = sum(
        len(seq) for seq in train_sequences.values()
    )

    train_items = {
        item
        for seq in train_sequences.values()
        for item in seq
    }

    print("\n  TRAIN DATA")
    print(f"  users        : {len(train_sequences):,}")
    print(f"  items        : {len(train_items):,}")
    print(f"  interactions : {train_interactions:,}")

    return train_sequences, train_items


def construct_cooccurrence(
    train_sequences,
    item_to_idx,
):
    """
    Build binary user-item matrix A and compute A.T @ A.

    Repeated appearances of an item in one user's sequence count once
    for co-occurrence.
    """

    print("\n[3/6] Building sparse user-item matrix...")

    rows = []
    cols = []

    for user_idx, (_, items) in enumerate(train_sequences.items()):
        unique_items = set(items)

        for item in unique_items:
            if item in item_to_idx:
                rows.append(user_idx)
                cols.append(item_to_idx[item])

    data = np.ones(len(rows), dtype=np.float32)

    A = sp.csr_matrix(
        (
            data,
            (
                np.asarray(rows),
                np.asarray(cols),
            ),
        ),
        shape=(
            len(train_sequences),
            len(item_to_idx),
        ),
        dtype=np.float32,
    )

    print(
        f"  A shape={A.shape}, "
        f"nnz={A.nnz:,}"
    )

    print("\n[4/6] Computing A.T @ A...")

    C = (A.T @ A).tocsr()

    # Remove self co-occurrence.
    C.setdiag(0)
    C.eliminate_zeros()

    print(f"  raw co-occurrence nnz={C.nnz:,}")

    return C


def build_mutual_topk(
    cooc,
    min_cooc=2,
    top_k=20,
):
    """
    Directed Top-K first, then retain only reciprocal neighbors.
    """

    print(
        f"\n[5/6] min_cooc={min_cooc}, "
        f"top_k={top_k}, symmetrize=mutual"
    )

    n_items = cooc.shape[0]

    directed_neighbors = {}

    candidate_directed = 0

    for i in range(n_items):
        start = cooc.indptr[i]
        end = cooc.indptr[i + 1]

        nbrs = cooc.indices[start:end]
        weights = cooc.data[start:end]

        mask = weights >= min_cooc
        nbrs = nbrs[mask]
        weights = weights[mask]

        candidate_directed += len(nbrs)

        if len(nbrs) == 0:
            directed_neighbors[i] = {}
            continue

        # Descending weight; node id as deterministic tie-breaker.
        order = np.lexsort((nbrs, -weights))

        if len(order) > top_k:
            order = order[:top_k]

        directed_neighbors[i] = {
            int(nbrs[j]): float(weights[j])
            for j in order
        }

    print(
        f"  candidate directed edges after min_cooc="
        f"{candidate_directed:,}"
    )

    edge_u = []
    edge_v = []
    edge_w = []

    for u in range(n_items):
        for v, weight_uv in directed_neighbors[u].items():
            if u >= v:
                continue

            reverse = directed_neighbors.get(v, {})

            if u in reverse:
                # A.T A is symmetric, but keep this robust.
                weight = min(weight_uv, reverse[u])

                edge_u.append(u)
                edge_v.append(v)
                edge_w.append(weight)

    return (
        np.asarray(edge_u, dtype=np.int64),
        np.asarray(edge_v, dtype=np.int64),
        np.asarray(edge_w, dtype=np.float32),
    )


def save_graph(
    output_dir,
    item_ids,
    edge_u,
    edge_v,
    edge_w,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(
        output_dir / "item_ids.npy",
        np.asarray(item_ids),
    )

    np.savez_compressed(
        output_dir / "edges.npz",
        src=edge_u,
        dst=edge_v,
        weight=edge_w,
    )

    n_nodes = len(item_ids)
    n_edges = len(edge_u)

    degree = np.zeros(n_nodes, dtype=np.int64)

    np.add.at(degree, edge_u, 1)
    np.add.at(degree, edge_v, 1)

    stats = {
        "nodes": int(n_nodes),
        "edges": int(n_edges),
        "avg_degree": float(
            degree.mean() if n_nodes else 0
        ),
        "max_degree": int(
            degree.max() if n_nodes else 0
        ),
        "isolated_nodes": int(
            np.sum(degree == 0)
        ),
        "mean_edge_weight": float(
            edge_w.mean() if n_edges else 0
        ),
        "min_edge_weight": float(
            edge_w.min() if n_edges else 0
        ),
        "max_edge_weight": float(
            edge_w.max() if n_edges else 0
        ),
    }

    with open(
        output_dir / "graph_stats.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            stats,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return stats


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--reviews", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument(
        "--min-user-interactions",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--min-cooc",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    metadata = load_valid_metadata(args.metadata)

    train_sequences, train_items = load_sequences(
        args.reviews,
        set(metadata),
        min_user_interactions=args.min_user_interactions,
    )

    # Fixed deterministic item ordering.
    item_ids = sorted(train_items)

    item_to_idx = {
        item: idx
        for idx, item in enumerate(item_ids)
    }

    print(
        f"\n  graph node universe="
        f"{len(item_ids):,}"
    )

    cooc = construct_cooccurrence(
        train_sequences,
        item_to_idx,
    )

    edge_u, edge_v, edge_w = build_mutual_topk(
        cooc,
        min_cooc=args.min_cooc,
        top_k=args.top_k,
    )

    print("\n[6/6] Saving graph...")

    stats = save_graph(
        args.output,
        item_ids,
        edge_u,
        edge_v,
        edge_w,
    )

    print("\n========== GRAPH COMPLETE ==========")

    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key:20s}: {value:.4f}")
        else:
            print(f"{key:20s}: {value:,}")

    print("\nSaved:")
    print(Path(args.output) / "edges.npz")
    print(Path(args.output) / "item_ids.npy")
    print(Path(args.output) / "graph_stats.json")


if __name__ == "__main__":
    main()
