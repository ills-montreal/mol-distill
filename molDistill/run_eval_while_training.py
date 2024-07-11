import os
import logging
import time

import argparse

import pandas as pd
import wandb

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# while true; launch the downstream_eval.py script if a new model is available in the folder specified
# by the MODEL_PATH variable that has not been seen before

ALL_DATASETS = [
    "hERG",
    "hERG_Karim",
    "AMES",
    "DILI",
    "Carcinogens_Lagunin",
    "Skin__Reaction",
    "Tox21",
    "ClinTox",
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
    "Caco2_Wang",
    "Lipophilicity_AstraZeneca",
    "Solubility_AqSolDB",
    "HydrationFreeEnergy_FreeSolv",
    "PPBR_AZ",
    "VDss_Lombardo",
    "Half_Life_Obach",
    "Clearance_Hepatocyte_AZ",
    "Clearance_Microsome_AZ",
    "LD50_Zhu",
    "HIV",
]


def launch_model_eval(model, MODEL_PATH):
    logger.info(f"Launching eval for {model}")
    if args.sbatch:
        os.system(f"sbatch eval.sh custom:{os.path.join(MODEL_PATH, model)} 5")
    else:
        os.system(
            f"python molDistill/downstream_eval.py --embedders custom:{os.path.join(MODEL_PATH, model)} --datasets hERG --test"
        )


def log_eval_results(model, MODEL_PATH):
    logger.info(f"Logging results for {model}")
    df = pd.read_csv(os.path.join(MODEL_PATH, model.replace(".pth", ".csv")))
    epoch = int(model.replace(".pth", "").split("_")[-1])
    df = df.drop(axis=1, columns=["Unnamed: 0", "embedder"])
    df = df.groupby(["dataset"]).mean().reset_index()
    all_logs = {
        f"eval_perfs_{dataset}": df[df.dataset == dataset].metric_test.values[0]
        for dataset in df.dataset.unique()
    }
    all_logs["eval_perfs"] = df.metric_test.mean()
    all_logs["epoch"] = epoch
    wandb.log(all_logs)


if __name__ == "__main__":
    wandb.init(
        project="mol-distill-downs-ckpt",
        allow_val_change=True,
    )
    for dataset in ALL_DATASETS:
        wandb.define_metric(f"eval_perfs_{dataset}", step_metric="epoch")
    wandb.define_metric("eval_perfs", step_metric="epoch")

    parser = argparse.ArgumentParser()
    parser.add_argument("MODEL_PATH", type=str)
    parser.add_argument("--sbatch", action="store_true")
    args = parser.parse_args()
    MODEL_PATH = args.MODEL_PATH
    checked_models = ["best_model.pth"]
    logged_models = ["best_model.pth"]
    last_round = False
    continue_training = True
    while continue_training:
        # Get all the models in the folder
        models = os.listdir(MODEL_PATH)
        models = [model for model in models if model.endswith(".pth")]
        # For every model
        for model in models:
            # If the model has not been checked
            if not model in checked_models:
                launch_model_eval(model, MODEL_PATH)
                checked_models.append(model)
            if (
                os.path.exists(os.path.join(MODEL_PATH, model.replace(".pth", ".csv")))
                and not model in logged_models
            ):
                log_eval_results(model, MODEL_PATH)
                logged_models.append(model)

        if (
            os.path.exists(os.path.join(MODEL_PATH, "stop.txt"))
            and logged_models == checked_models
        ):
            if last_round:
                continue_training = False
            last_round = True

        logger.info("Sleeping for 10 seconds")
        time.sleep(10)

    checked_models = checked_models[1:]

    logger.info("Finished logging all results")

    df = pd.concat(
        [
            pd.read_csv(os.path.join(MODEL_PATH, model.replace(".pth", ".csv")))
            for model in checked_models
        ]
    )
    df["epoch"] = df.embedder.apply(lambda x: int(x.replace(".pth", "").split("_")[-1]))
    df = df.drop(axis=1, columns=["Unnamed: 0", "embedder"])
    df = df.groupby(["epoch", "dataset"]).mean().reset_index()
    wandb.log({"table": wandb.Table(dataframe=df)})

    wandb.finish()
