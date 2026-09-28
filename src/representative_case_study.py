import argparse
import gzip
import ast
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


def load_metadata(meta_path):
    titles = {}

    with gzip.open(meta_path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                obj = ast.literal_eval(line.strip())
            except Exception:
                continue

            asin = obj.get("asin")
            title = obj.get("title")

            if asin and title:
                titles[str(asin)] = str(title)

    return titles


def load_lcc_edges(graph_dir):
    edge_data = np.load(graph_dir / "edges.npz")

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


def compute_global_scores(df, src, dst, weight):
    g = df["global_label"].to_numpy()

    stats = defaultdict(
        lambda: {
            "nodes": 0,
            "fine_internal_edges": 0,
            "fine_internal_weight": 0.0,
        }
    )

    global_counts = df["global_label"].value_counts()

    for gid, count in global_counts.items():
        stats[int(gid)]["nodes"] = int(count)

    local = df["local_label"].to_numpy()
    fine = df["fine_label"].to_numpy()

    for s, d, w in zip(src, dst, weight):
        if g[s] != g[d]:
            continue

        if (
            local[s] == local[d]
            and fine[s] == fine[d]
        ):
            gid = int(g[s])

            stats[gid]["fine_internal_edges"] += 1
            stats[gid]["fine_internal_weight"] += float(w)

    rows = []

    for gid, s in stats.items():
        nodes = s["nodes"]

        rows.append(
            {
                "global_label": gid,
                "nodes": nodes,
                "fine_internal_edges": s[
                    "fine_internal_edges"
                ],
                "fine_internal_weight": s[
                    "fine_internal_weight"
                ],
                "score": (
                    s["fine_internal_edges"]
                    / max(nodes, 1)
                ),
            }
        )

    return pd.DataFrame(rows)


def select_cases(score_df, seed):
    score_df = score_df.sort_values(
        "score"
    ).reset_index(drop=True)

    strong_row = score_df.iloc[-1]

    median_idx = len(score_df) // 2
    median_row = score_df.iloc[median_idx]

    rng = np.random.default_rng(seed)

    excluded = {
        int(strong_row["global_label"]),
        int(median_row["global_label"]),
    }

    candidates = score_df[
        ~score_df["global_label"].isin(excluded)
    ]

    random_row = candidates.iloc[
        rng.integers(0, len(candidates))
    ]

    return {
        "strong": strong_row,
        "median": median_row,
        "random": random_row,
    }


def build_strength_map(
    df,
    src,
    dst,
    weight,
):
    global_arr = df[
        "global_label"
    ].to_numpy()

    local_arr = df[
        "local_label"
    ].to_numpy()

    fine_arr = df[
        "fine_label"
    ].to_numpy()

    strength = defaultdict(float)

    for s, d, w in zip(src, dst, weight):
        same_fine = (
            global_arr[s] == global_arr[d]
            and local_arr[s] == local_arr[d]
            and fine_arr[s] == fine_arr[d]
        )

        if same_fine:
            strength[int(s)] += float(w)
            strength[int(d)] += float(w)

    return strength


def write_case(
    f,
    case_name,
    row,
    df,
    titles,
    strength,
    max_local_groups,
    max_fine_groups,
    max_items,
):
    gid = int(row["global_label"])

    subset = df[
        df["global_label"] == gid
    ].copy()

    f.write(
        f"\n{'=' * 80}\n"
    )
    f.write(
        f"{case_name.upper()} CASE\n"
    )
    f.write(
        f"{'=' * 80}\n"
    )

    f.write(
        f"Global label: {gid}\n"
    )
    f.write(
        f"Items: {len(subset)}\n"
    )
    f.write(
        f"Selection score: "
        f"{float(row['score']):.6f}\n"
    )
    f.write(
        f"Fine internal edges: "
        f"{int(row['fine_internal_edges'])}\n"
    )

    local_sizes = (
        subset
        .groupby("local_label")
        .size()
        .sort_values(ascending=False)
    )

    selected_local = (
        local_sizes
        .head(max_local_groups)
        .index
        .tolist()
    )

    for lid in selected_local:
        local_df = subset[
            subset["local_label"] == lid
        ]

        f.write(
            f"\n--- Local {lid} "
            f"(items={len(local_df)}) ---\n"
        )

        fine_sizes = (
            local_df
            .groupby("fine_label")
            .size()
            .sort_values(ascending=False)
        )

        selected_fine = (
            fine_sizes
            .head(max_fine_groups)
            .index
            .tolist()
        )

        for fid in selected_fine:
            fine_df = local_df[
                local_df["fine_label"] == fid
            ].copy()

            fine_df["graph_strength"] = [
                strength.get(int(idx), 0.0)
                for idx in fine_df.index
            ]

            fine_df = fine_df.sort_values(
                "graph_strength",
                ascending=False,
            )

            f.write(
                f"\n  Fine {fid} "
                f"(items={len(fine_df)})\n"
            )

            for idx, item in fine_df.head(
                max_items
            ).iterrows():
                item_id = str(item["item_id"])
                title = titles.get(
                    item_id,
                    "[TITLE NOT FOUND]",
                )

                f.write(
                    f"    - "
                    f"{item_id} | "
                    f"strength="
                    f"{strength.get(int(idx), 0.0):.1f}"
                    f" | {title}\n"
                )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--graph-dir",
        required=True,
    )

    parser.add_argument(
        "--meta-path",
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--max-local-groups",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--max-fine-groups",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--max-items",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    graph_dir = Path(
        args.graph_dir
    )

    hierarchy_path = (
        graph_dir
        / "hierarchical_labels.csv"
    )

    print("[1/5] Loading hierarchy...")

    df = pd.read_csv(
        hierarchy_path
    )

    # row order must correspond to LCC-local index
    df.index = np.arange(
        len(df)
    )

    print(
        f"items : {len(df):,}"
    )

    print("[2/5] Loading graph...")

    src, dst, weight = load_lcc_edges(
        graph_dir
    )

    print(
        f"LCC edges : {len(src):,}"
    )

    print("[3/5] Selecting representative cases...")

    score_df = compute_global_scores(
        df,
        src,
        dst,
        weight,
    )

    cases = select_cases(
        score_df,
        seed=args.seed,
    )

    for name, row in cases.items():
        print(
            f"{name:6s} | "
            f"global={int(row['global_label']):2d} | "
            f"items={int(row['nodes']):4d} | "
            f"score={float(row['score']):.4f}"
        )

    print("[4/5] Loading metadata...")

    titles = load_metadata(
        args.meta_path
    )

    print(
        f"metadata titles : {len(titles):,}"
    )

    strength = build_strength_map(
        df,
        src,
        dst,
        weight,
    )

    print("[5/5] Writing case study...")

    output_path = (
        graph_dir
        / "representative_case_study.txt"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "Representative Hierarchical Pseudo-label Case Study\n"
        )

        f.write(
            "Cases: strong / median / random\n"
        )

        f.write(
            "graph_strength = sum of edge weights "
            "to items inside the same Fine cluster\n"
        )

        for case_name in [
            "strong",
            "median",
            "random",
        ]:
            write_case(
                f,
                case_name,
                cases[case_name],
                df,
                titles,
                strength,
                args.max_local_groups,
                args.max_fine_groups,
                args.max_items,
            )

    score_output = (
        graph_dir
        / "global_case_scores.csv"
    )

    score_df.to_csv(
        score_output,
        index=False,
    )

    print("\nSaved:")
    print(output_path)
    print(score_output)


if __name__ == "__main__":
    main()
