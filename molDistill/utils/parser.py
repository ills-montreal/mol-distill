import argparse


GROUPED_MODELS = {
    "GNN": [
        # "ContextPred",
        # "GPT-GNN",
        "GraphMVP",
        # "GROVER",
        # "AttributeMask",
        "GraphLog",
        "GraphCL",
        # "InfoGraph",
    ],
    "BERT": [
        # "MolBert",
        # "ChemBertMLM-5M",
        # "ChemBertMLM-10M",
        # "ChemBertMLM-77M",
        "ChemBertMTR-5M",
        "ChemBertMTR-10M",
        "ChemBertMTR-77M",
    ],
    "GPT": [
        # "ChemGPT-1.2B",
        # "ChemGPT-19M",
        # "ChemGPT-4.7M",
    ],
    "Denoising": [
        "DenoisingPretrainingPQCMv4",
        "FRAD_QM9",
    ],
    "MolR": [
        "MolR_gat",
        # "MolR_gcn",
        "MolR_tag",
    ],
    "MoleOOD": [
        # "MoleOOD_OGB_GIN",
        # "MoleOOD_OGB_GCN",
        # "MoleOOD_OGB_SAGE",
    ],
    "ThreeD": [
        "ThreeDInfomax",
    ],
}


def update_grouped_models(embedders):
    new_embedders = []
    for embedder in embedders:
        if embedder in GROUPED_MODELS:
            new_embedders += GROUPED_MODELS[embedder]
        else:
            new_embedders.append(embedder)
    new_embedders = list(set(new_embedders))
    return new_embedders


def get_pretraining_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="../data")
    parser.add_argument("--dataset", type=str, default="MOSES")

    parser.add_argument(
        "--embedders-to-simulate",
        nargs="+",
        type=str,
        default=["GNN", "BERT", "GPT", "Denoising", "ThreeD"],
    )

    # training parameters
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--valid-prop", type=float, default=0.1)

    parser.add_argument(
        "--gnn-type",
        type=str,
        default="gine",
        choices=["gin", "gine", "gcn", "gat", "graphsage", "tag", "arma", "gatv2"],
    )
    parser.add_argument(
        "--node-embedding-keys",
        nargs="+",
        type=str,
        default=["possible_atomic_num_list", "possible_chirality_list"],
    )
    parser.add_argument("--n-layer", type=int, default=10)
    parser.add_argument("--n-MLP-layer", type=int, default=1)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--drop-ratio", type=float, default=0.0)
    parser.add_argument("--batch-norm-type", type=str, default="layer")
    parser.add_argument("--graph-pooling", type=str, default="mean")

    # other parameters
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="../results")
    parser.add_argument("--save-name", type=str, default="tmp")
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--checkpoint", type=str, default=None)

    parser.add_argument("--n-workers", type=int, default=6)

    return parser
