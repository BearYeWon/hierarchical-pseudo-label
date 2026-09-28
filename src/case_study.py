import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def load_titles(metadata_path, target_items):
    target_items = set(map(str, target_items))
    titles = {}

    print("[1/4] Loading item titles...")

    with gzip.open(
        metadata_path,
        "rt",
        encoding="utf-8",
    ) as f:
        for i, line in enumerate(f, 1):
            obj = json.loads(line)

            asin = str(
                obj.get("asin") or ""
            )

            if asin not in target_items:
                continue

            title = str(
                obj.get("title") or ""
            ).strip()

            if title:
                titles[asin] = title

            if len(titles) == len(target_items):
                break

    print(
        f"titles found : "
        f"{len(titles):,} / {len(target_items):,}"
    )

    return titles


def build_lcc_edges(graph_dir):
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


def compute_internal_edges(
    labels_df,
    src,
    dst,
):
    fine_edge_counts = defaultdict(int)

    g = labels_df[
        "global_label"
    ].to_numpy()

    l = labels_df[
        "local_label"
    ].to_numpy()

    f = labels_df[
        "fine_label"
    ].to_numpy()

    for u, v in zip(src, dst):
        if (
            g[u] == g[v]
            and l[u] == l[v]
            and f[u] == f[v]
        ):
            key = (
                int(g[u]),
                int(l[u]),
                int(f[u]),
            )
            fine_edge_counts[key] += 1

    return fine_edge_counts


def select_global_cluster(
    df,
    fine_edge_counts,
):
    """
    Pick a useful case-study Global cluster:
    - multiple local/fine groups
    - sufficiently many internal Fine edges
    """

    scores = defaultdict(int)

    for (
        g,
        l,
        f,
    ), count in fine_edge_counts.items():
        scores[g] += count

    candidates = []

    for g, group in df.groupby(
        "global_label"
    ):
        num_local = group[
            "local_label"
        ].nunique()

        num_fine = (
            group[
                [
                    "local_label",
                    "fine_label",
                ]
            ]
            .drop_duplicates()
            .shape[0]
        )

        score = scores.get(
            int(g),
            0,
        )

        candidates.append(
            (
                score,
                num_fine,
                num_local,
                len(group),
                int(g),
            )
        )

    candidates.sort(
        reverse=True
    )

    return candidates[0][-1]


def item_degree_inside_fine(
    indices,
    src,
    dst,
    weight,
):
    target = set(
        map(int, indices)
    )

    score = defaultdict(float)

    for u, v, w in zip(
        src,
        dst,
        weight,
    ):
        u = int(u)
        v = int(v)

        if (
            u in target
            and v in target
        ):
            score[u] += float(w)
            score[v] += float(w)

    return score


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--graph-dir",
        required=True,
    )

    parser.add_argument(
        "--metadata",
        required=True,
    )

    parser.add_argument(
        "--global-id",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-local",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--max-fine",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--items-per-fine",
        type=int,
        default=6,
    )

    args = parser.parse_args()

    graph_dir = Path(
        args.graph_dir
    )

    print(
        "[1/4] Loading hierarchy..."
    )

    df = pd.read_csv(
        graph_dir
        / "hierarchical_labels.csv",
        dtype={"item_id": str},
    )

    (
        src,
        dst,
        weight,
    ) = build_lcc_edges(
        graph_dir
    )

    fine_edge_counts = (
        compute_internal_edges(
            df,
            src,
            dst,
        )
    )

    if args.global_id is None:
        global_id = (
            select_global_cluster(
                df,
                fine_edge_counts,
            )
        )
    else:
        global_id = args.global_id

    global_df = df[
        df["global_label"]
        == global_id
    ].copy()

    print(
        f"selected Global "
        f"{global_id}"
    )
    print(
        f"items : "
        f"{len(global_df):,}"
    )

    print(
        "\n[2/4] Loading metadata..."
    )

    titles = load_titles(
        args.metadata,
        global_df["item_id"],
    )

    print(
        "\n[3/4] Building hierarchy case study..."
    )

    lines = []

    lines.append(
        "=" * 70
    )
    lines.append(
        "HIERARCHICAL GRAPH PSEUDO-LABEL CASE STUDY"
    )
    lines.append(
        "=" * 70
    )
    lines.append(
        f"Global cluster: {global_id}"
    )
    lines.append(
        f"Items: {len(global_df)}"
    )
    lines.append("")

    local_sizes = (
        global_df
        .groupby("local_label")
        .size()
        .sort_values(
            ascending=False
        )
    )

    selected_locals = (
        local_sizes
        .head(args.max_local)
        .index
    )

    for local_id in selected_locals:

        local_df = global_df[
            global_df["local_label"]
            == local_id
        ]

        lines.append(
            f"├── Local {local_id} "
            f"(n={len(local_df)})"
        )

        fine_sizes = (
            local_df
            .groupby("fine_label")
            .size()
            .sort_values(
                ascending=False
            )
        )

        selected_fines = (
            fine_sizes
            .head(args.max_fine)
            .index
        )

        for fine_id in selected_fines:

            fine_df = local_df[
                local_df["fine_label"]
                == fine_id
            ]

            indices = (
                fine_df.index
                .to_numpy()
            )

            scores = (
                item_degree_inside_fine(
                    indices,
                    src,
                    dst,
                    weight,
                )
            )

            ranked_indices = sorted(
                indices,
                key=lambda x: scores.get(
                    int(x),
                    0.0,
                ),
                reverse=True,
            )

            ranked_indices = (
                ranked_indices[
                    :args.items_per_fine
                ]
            )

            edge_count = (
                fine_edge_counts.get(
                    (
                        int(global_id),
                        int(local_id),
                        int(fine_id),
                    ),
                    0,
                )
            )

            lines.append(
                f"│   ├── Fine {fine_id} "
                f"(n={len(fine_df)}, "
                f"internal_edges={edge_count})"
            )

            for idx in ranked_indices:

                row = df.loc[idx]

                asin = str(
                    row["item_id"]
                )

                title = titles.get(
                    asin,
                    "[title unavailable]",
                )

                score = scores.get(
                    int(idx),
                    0.0,
                )

                lines.append(
                    f"│   │   • {title}"
                    f"  [{asin}]"
                    f"  graph_strength={score:.1f}"
                )

            lines.append(
                "│   │"
            )

        lines.append("│")

    output_text = "\n".join(
        lines
    )

    print(
        "\n[4/4] Case study\n"
    )

    print(output_text)

    output_path = (
        graph_dir
        / "case_study.txt"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(output_text)

    print(
        f"\nSaved: {output_path}"
    )


if __name__ == "__main__":
    main()
