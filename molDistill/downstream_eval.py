import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import argparse
import json
import logging
from typing import Dict, List, Callable, Optional, Tuple
from functools import partial

import datamol as dm
import pandas as pd
import torch
import wandb
from molDistill.baselines.utils import MolecularFeatureExtractor
from molDistill.baselines.utils.evaluation import (
    get_dataloaders,
    Feed_forward,
    FFConfig,
    FF_trainer,
)
from molDistill.baselines.utils.tdc_dataset import EvaluationDatasetIterable
from tqdm import tqdm

DATASETS_GROUP = {
    "TOX": [
        "hERG",
        "hERG_Karim",
        "AMES",
        "DILI",
        "Carcinogens_Lagunin",
        "Skin__Reaction",
        "Tox21",
        "ClinTox",
        # "ToxCast",
    ],
    "ADME": [
        "PAMPA_NCATS",
        "HIA_Hou",
        "Pgp_Broccatelli",
        "Bioavailability_Ma",
        "BBB_Martins",
        "CYP2C19_Veith",
        "CYP2D6_Veith",
        "CYP3A4_Veith",
        "CYP1A2_Veith",
        "CYP2C9_Veith",
        "CYP2C9_Substrate_CarbonMangels",
        "CYP2D6_Substrate_CarbonMangels",
        "CYP3A4_Substrate_CarbonMangels",
    ],
    "ADME_REG": [
        "Caco2_Wang",
        "Lipophilicity_AstraZeneca",
        "Solubility_AqSolDB",
        "HydrationFreeEnergy_FreeSolv",
        "PPBR_AZ",
        "VDss_Lombardo",
        "Half_Life_Obach",
        "Clearance_Hepatocyte_AZ",
        "Clearance_Microsome_AZ",
    ],
    "TOX_REG": [
        "LD50_Zhu",
    ],
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODELS = [
    # "ContextPred",
    # "GPT-GNN",
    "GraphMVP",
    "GROVER",
    # "AttributeMask",
    "GraphLog",
    "GraphCL",
    "InfoGraph",
    # "Not-trained",
    # "MolBert",
    # "ChemBertMLM-5M",
    "ChemBertMLM-10M",
    # "ChemBertMLM-77M",
    # "ChemBertMTR-5M",
    # "ChemBertMTR-10M",
    "ChemBertMTR-77M",
    "ChemGPT-1.2B",
    # "ChemGPT-19M",
    # "ChemGPT-4.7M",
    # "DenoisingPretrainingPQCMv4",
    "FRAD_QM9",
    "MolR_gat",
    # "MolR_gcn",
    # "MolR_tag",
    # "MoleOOD_OGB_GIN",
    # "MoleOOD_OGB_GCN",
    "ThreeDInfomax",
    "custom:MOSES_512_10_lr1e-4_gine/model_95.pth",
]


def get_all_embs(dataset: str, args: argparse.Namespace, data_path: str = "../data"):
    with open(os.path.join(data_path, "smiles.json"), "r") as f:
        smiles = json.load(f)
    mol_path = os.path.join(data_path, "preprocessed.sdf")
    if os.path.exists(mol_path):
        mols = dm.read_sdf(mol_path)
    else:
        mols = dm.to_mol(smiles)
    feature_extractor = MolecularFeatureExtractor(
        dataset=dataset,
        device=args.device,
        data_dir=args.data_path,
    )
    embeddings = {
        model_name: feature_extractor.get_features(
            smiles=smiles,
            mols=mols,
            name=model_name,
        )
        for model_name in args.embedders
    }
    return smiles, mols, embeddings


def get_split_emb(
    split_idx: Dict[str, List[int]],
    embeddings: Dict[str, torch.Tensor],
    smiles: List[str],
    mols: List[dm.Mol],
    embedder_name: str = "ecfp",
):
    X = embeddings[embedder_name]
    split_emb = {}
    for key in split_idx.keys():
        split_emb[key] = {
            "x": X[split_idx[key]["x"]].to("cpu"),
            "y": torch.tensor(split_idx[key]["y"]),
        }
    return split_emb


def launch_evaluation(
    dataset: str,
    embedder_name: str,
    device: str,
    split_idx: Dict[str, List[int]],
    model_config: Dict,
    smiles: List[str],
    mols: List[dm.Mol],
    embeddings: Dict[str, torch.Tensor],
    plot_loss: bool = False,
    run_num: Optional[Tuple[int, int]] = None,
    test: bool = False,
):
    split_emb = get_split_emb(split_idx, embeddings, smiles, mols, embedder_name)

    dataloader_train, dataloader_val, dataloader_test, input_dim = get_dataloaders(
        split_emb,
        batch_size=model_config.batch_size,
        test_batch_size=model_config.test_batch_size,
    )
    y_train = split_emb["train"]["y"]
    if len(y_train.shape) == 1:
        output_dim = 1
    else:
        output_dim = split_emb["train"]["y"].shape[1]
    if dataset in DATASETS_GROUP["ADME_REG"] + DATASETS_GROUP["TOX_REG"]:
        model = Feed_forward(
            input_dim,
            output_dim,
            model_config,
            task="regression",
            task_size=y_train.shape[0],
            out_train_mean=y_train.mean(),
            out_train_std=y_train.std(),
        )
    else:
        model = Feed_forward(
            input_dim,
            output_dim,
            model_config,
            task="classification",
            task_size=y_train.shape[0],
        )

    trainer = FF_trainer(model)

    trainer.train_model(
        dataloader_train,
        dataloader_val,
        p_bar_name=f"{run_num[0]} / {run_num[1]} : \t{dataset} - {embedder_name} : ",
    )

    model = trainer.model
    if args.wandb:
        for val_roc in model.val_roc:
            wandb.log({f"val_roc": val_roc})
        for r2 in model.r2_val:
            wandb.log({f"r2_val": r2})

    if test:
        test_metric = trainer.eval_on_test(dataloader_test)
    else:
        test_metric = 0

    df_results = pd.DataFrame(
        {
            "embedder": [embedder_name],
            "dataset": [dataset],
            "metric_test": test_metric,
            "metric": trainer.best_metric,
        }
    )
    return df_results


def main(args):
    final_res = []

    for dataset in args.datasets:
        data_path = os.path.join(args.data_path, dataset)
        model_config = FFConfig(
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            d_rate=args.d_rate,
            norm=args.norm,
            lr=args.lr,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            device=args.device,
            test_batch_size=args.test_batch_size,
        )

        smiles, mols, embeddings = get_all_embs(dataset, args, data_path)

        for random_seed in range(args.n_runs):
            print(f"Dataset {dataset} - Seed {random_seed}")
            try:
                split_sampler = EvaluationDatasetIterable(
                    dataset=dataset,
                    random_seed=random_seed,
                    method=args.split_method,
                    smiles=smiles,
                )
            except Exception as e:
                logger.error(f"Error in dataset {dataset} with seed {random_seed}")
                logger.error(e)
                raise e
                continue

            for i_split, split_idx in enumerate(
                tqdm(
                    split_sampler.sample(),
                    desc=f"Dataset {dataset} - Seed {random_seed}",
                    total=len(split_sampler.label_list),
                )
            ):
                for i, embedder_name in enumerate(args.embedders):
                    res = launch_evaluation(
                        dataset=dataset,
                        embedder_name=embedder_name,
                        split_idx=split_idx,
                        device=args.device,
                        model_config=model_config,
                        smiles=smiles,
                        mols=mols,
                        embeddings=embeddings,
                        plot_loss=args.plot_loss,
                        run_num=(i, len(args.embedders)),
                        test=args.test,
                    )
                    final_res.append(res)

    df = pd.concat(final_res).reset_index(drop=True)
    if args.wandb:
        wandb.log({"results_df": wandb.Table(dataframe=df)})
    df_grouped = df.groupby(["embedder", "dataset"]).mean().reset_index()

    if args.wandb:
        wandb.log(
            {"mean_metric": df_grouped.groupby("embedder")["metric"].mean().mean()}
        )

    if len(args.embedders) == 1 and args.embedders[0].startswith("custom:"):
        path = args.embedders[0].split(":")[1]
        df_grouped.to_csv(path.replace(".pth", ".csv"))
    if args.save_results:
        for dataset in args.datasets:
            for embedder in args.embedders:
                os.makedirs(
                    f"downstream_results/{embedder.split('/')[-1].replace('.pth','')}", exist_ok=True
                )
                df[(df["dataset"] == dataset) & (df["embedder"] == embedder)].to_csv(
                    f"downstream_results/{embedder.split('/')[-1].replace('.pth','')}/results_{dataset}.csv"
                )


def add_downstream_args(parser: argparse.ArgumentParser):
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--d-rate", type=float, default=0.0)
    parser.add_argument("--norm", type=str, default="layer")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--save-results", action="store_true")

    parser.add_argument(
        "--data-path",
        type=str,
        default="../data",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["TOX", "ADME", "HIV", "ADME_REG", "TOX_REG"],
    )
    parser.add_argument(
        "--embedders",
        type=str,
        nargs="+",
        default=None,
        required=False,
        help="Embedders to use",
    )
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--plot-loss", action="store_true")
    parser.add_argument("--split-method", type=str, default="random")

    parser.add_argument("--test", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch the evaluation of a downstream model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser = add_downstream_args(parser)

    args = parser.parse_args()

    if args.embedders is None:
        args.embedders = MODELS

    for group in DATASETS_GROUP.keys():
        if group in args.datasets:
            args.datasets.remove(group)
            args.datasets += DATASETS_GROUP[group]

    if args.wandb:
        if args.split_method == "scaffold":
            wandb.init(project="distill-downstream", tags=["scaffold"])
        else:
            wandb.init(project="distill-downstream")

        wandb.config.update(args)
    main(args)
