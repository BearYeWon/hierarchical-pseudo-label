import argparse
import random
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# Graph loading
# ---------------------------------------------------------

def load_graph(graph_dir):
    graph_dir = Path(graph_dir)

    edge_data = np.load(graph_dir / "edges.npz")

    src = edge_data["src"].astype(np.int64)
    dst = edge_data["dst"].astype(np.int64)
    weight = edge_data["weight"].astype(np.float32)

    item_ids = np.load(
        graph_dir / "item_ids.npy",
        allow_pickle=True,
    )

    lcc_nodes = np.load(
        graph_dir / "lcc_nodes.npy"
    ).astype(np.int64)

    return item_ids, src, dst, weight, lcc_nodes


def build_lcc_graph(
    num_nodes,
    src,
    dst,
    weight,
    lcc_nodes,
):
    """
    Convert original graph indices -> LCC-local indices.
    """

    old_to_new = np.full(
        num_nodes,
        -1,
        dtype=np.int64,
    )

    old_to_new[lcc_nodes] = np.arange(
        len(lcc_nodes)
    )

    src_new = old_to_new[src]
    dst_new = old_to_new[dst]

    mask = (
        (src_new >= 0)
        & (dst_new >= 0)
    )

    src_new = src_new[mask]
    dst_new = dst_new[mask]
    weight_new = weight[mask]

    return (
        src_new,
        dst_new,
        weight_new,
    )


# ---------------------------------------------------------
# LightGCN normalized adjacency
# ---------------------------------------------------------

def build_normalized_adj(
    num_nodes,
    src,
    dst,
    weight,
    device,
):
    """
    Weighted symmetric normalized adjacency:

        S = D^{-1/2} A D^{-1/2}
    """

    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    values = np.concatenate([weight, weight])

    adj = sp.coo_matrix(
        (values, (rows, cols)),
        shape=(num_nodes, num_nodes),
        dtype=np.float32,
    ).tocsr()

    degree = np.asarray(
        adj.sum(axis=1)
    ).reshape(-1)

    degree_inv_sqrt = np.zeros_like(
        degree,
        dtype=np.float32,
    )

    nonzero = degree > 0

    degree_inv_sqrt[nonzero] = (
        1.0 / np.sqrt(degree[nonzero])
    )

    norm_values = (
        values
        * degree_inv_sqrt[rows]
        * degree_inv_sqrt[cols]
    )

    indices = torch.tensor(
        np.vstack([rows, cols]),
        dtype=torch.long,
        device=device,
    )

    values_t = torch.tensor(
        norm_values,
        dtype=torch.float32,
        device=device,
    )

    norm_adj = torch.sparse_coo_tensor(
        indices,
        values_t,
        size=(num_nodes, num_nodes),
        device=device,
    ).coalesce()

    return norm_adj


# ---------------------------------------------------------
# LightGCN
# ---------------------------------------------------------

class LightGCN(nn.Module):

    def __init__(
        self,
        num_nodes,
        embedding_dim=128,
        num_layers=3,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            num_nodes,
            embedding_dim,
        )

        nn.init.normal_(
            self.embedding.weight,
            std=0.1,
        )

    def forward(self, norm_adj):
        """
        E^(0) = learnable random item embedding

        E^(k+1) = S E^(k)

        final =
        mean(E^0, E^1, ..., E^K)
        """

        x = self.embedding.weight

        embeddings = [x]

        for _ in range(self.num_layers):
            x = torch.sparse.mm(
                norm_adj,
                x,
            )

            embeddings.append(x)

        out = torch.stack(
            embeddings,
            dim=0,
        ).mean(dim=0)

        return out


# ---------------------------------------------------------
# Negative sampling
# ---------------------------------------------------------

def build_neighbor_sets(
    num_nodes,
    src,
    dst,
):
    neighbors = [
        set()
        for _ in range(num_nodes)
    ]

    for u, v in zip(src, dst):
        u = int(u)
        v = int(v)

        neighbors[u].add(v)
        neighbors[v].add(u)

    return neighbors


def sample_negative(
    u,
    num_nodes,
    neighbors,
):
    while True:
        neg = random.randrange(
            num_nodes
        )

        if (
            neg != u
            and neg not in neighbors[u]
        ):
            return neg


# ---------------------------------------------------------
# BPR training
# ---------------------------------------------------------

def train_lightgcn(
    model,
    norm_adj,
    src,
    dst,
    num_nodes,
    epochs,
    batch_size,
    lr,
    reg,
    device,
):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    neighbors = build_neighbor_sets(
        num_nodes,
        src,
        dst,
    )

    edges = np.stack(
        [src, dst],
        axis=1,
    )

    n_edges = len(edges)

    print(
        f"training edges : {n_edges:,}"
    )

    for epoch in range(
        1,
        epochs + 1,
    ):
        np.random.shuffle(edges)

        epoch_loss = 0.0
        num_batches = 0

        for start in range(
            0,
            n_edges,
            batch_size,
        ):
            batch = edges[
                start:start + batch_size
            ]

            if len(batch) == 0:
                continue

            users = []
            positives = []
            negatives = []

            for u, v in batch:
                # Randomly flip direction.
                if random.random() < 0.5:
                    u, v = v, u

                users.append(u)
                positives.append(v)

                negatives.append(
                    sample_negative(
                        int(u),
                        num_nodes,
                        neighbors,
                    )
                )

            u = torch.tensor(
                users,
                dtype=torch.long,
                device=device,
            )

            pos = torch.tensor(
                positives,
                dtype=torch.long,
                device=device,
            )

            neg = torch.tensor(
                negatives,
                dtype=torch.long,
                device=device,
            )

            optimizer.zero_grad()

            z = model(norm_adj)

            z_u = z[u]
            z_pos = z[pos]
            z_neg = z[neg]

            pos_score = (
                z_u * z_pos
            ).sum(dim=1)

            neg_score = (
                z_u * z_neg
            ).sum(dim=1)

            bpr_loss = -F.logsigmoid(
                pos_score - neg_score
            ).mean()

            # Regularize initial trainable embeddings.
            e0 = model.embedding.weight

            reg_loss = (
                e0[u].pow(2).sum()
                + e0[pos].pow(2).sum()
                + e0[neg].pow(2).sum()
            ) / len(batch)

            loss = (
                bpr_loss
                + reg * reg_loss
            )

            loss.backward()
            optimizer.step()

            epoch_loss += (
                loss.item()
            )

            num_batches += 1

        print(
            f"[epoch {epoch:03d}] "
            f"loss="
            f"{epoch_loss / max(num_batches, 1):.6f}"
        )

    return model


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--graph-dir",
        type=str,
        default="outputs/toys_behavior_mutual",
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default="structural_embeddings.npy",
    )

    parser.add_argument(
        "--dim",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--layers",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--reg",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            args.seed
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"device : {device}")

    graph_dir = Path(
        args.graph_dir
    )

    print("\n[1/5] Loading graph...")

    (
        item_ids,
        src,
        dst,
        weight,
        lcc_nodes,
    ) = load_graph(
        graph_dir
    )

    print(
        f"original nodes : "
        f"{len(item_ids):,}"
    )

    print(
        f"LCC nodes      : "
        f"{len(lcc_nodes):,}"
    )

    print("\n[2/5] Building LCC graph...")

    (
        lcc_src,
        lcc_dst,
        lcc_weight,
    ) = build_lcc_graph(
        len(item_ids),
        src,
        dst,
        weight,
        lcc_nodes,
    )

    num_nodes = len(
        lcc_nodes
    )

    print(
        f"LCC edges : "
        f"{len(lcc_src):,}"
    )

    print("\n[3/5] Building normalized adjacency...")

    norm_adj = build_normalized_adj(
        num_nodes,
        lcc_src,
        lcc_dst,
        lcc_weight,
        device,
    )

    model = LightGCN(
        num_nodes=num_nodes,
        embedding_dim=args.dim,
        num_layers=args.layers,
    ).to(device)

    print("\n[4/5] Training LightGCN...")

    model = train_lightgcn(
        model,
        norm_adj,
        lcc_src,
        lcc_dst,
        num_nodes,
        args.epochs,
        args.batch_size,
        args.lr,
        args.reg,
        device,
    )

    print("\n[5/5] Extracting embeddings...")

    model.eval()

    with torch.no_grad():
        embeddings = model(
            norm_adj
        )

        embeddings = F.normalize(
            embeddings,
            p=2,
            dim=1,
        )

    embeddings = (
        embeddings
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    output_path = (
        graph_dir
        / args.output_name
    )

    np.save(
        output_path,
        embeddings,
    )

    print("\n========== COMPLETE ==========")

    print(
        f"embedding shape : "
        f"{embeddings.shape}"
    )

    print(
        f"mean norm       : "
        f"{np.linalg.norm(embeddings, axis=1).mean():.4f}"
    )

    print()

    print(
        f"saved : {output_path}"
    )


if __name__ == "__main__":
    main()
