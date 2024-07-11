import os
import logging
import time

import argparse

import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# while true; launch the downstream_eval.py script if a new model is available in the folder specified
# by the MODEL_PATH variable that has not been seen before

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("MODEL_PATH", type=str)
    parser.add_argument("--sbatch", action="store_true")
    args = parser.parse_args()
    MODEL_PATH = args.MODEL_PATH
    checked_models = ["best_model.pth"]
    last_round = False
    continue_training = True
    while continue_training:
        # Get all the models in the folder
        models = os.listdir(MODEL_PATH)
        # For every model
        for model in models:
            # If the model has not been checked
            if model.endswith(".pth") and not model in checked_models:
                # Launch the downstream eval script
                logger.info(f"Launching eval for {model}")
                if args.sbatch:
                    os.system(
                        f"sbatch eval.sh custom:{os.path.join(MODEL_PATH, model)} 5"
                    )
                else:
                    os.system(
                        f"python molDistill/downstream_eval.py --embedders custom:{os.path.join(MODEL_PATH, model)} --datasets hERG"
                    )
                # Add the model to the checked models
                checked_models.append(model)

            # if stop.txt is in the directory, stop the script
            if model == "stop.txt":
                if last_round:
                    continue_training = False
                last_round = True

        logger.info("Sleeping for 10 seconds")
        time.sleep(10)

    checked_models = checked_models[1:]

    logger.info("Finished logging all results")

    all_csv_exist = False
    while not all_csv_exist:
        all_csv_exist = all(
            os.path.exists(os.path.join(MODEL_PATH, model.replace(".pth", ".csv")))
            for model in checked_models
        )
        time.sleep(10)

    import wandb

    wandb.init(
        project="mol-distill-downs-ckpt",
        allow_val_change=True,
    )

    df = pd.concat(
        [
            pd.read_csv(os.path.join(MODEL_PATH, model.replace(".pth", ".csv")))
            for model in checked_models
        ]
    )
    df["epoch"] = df.embedder.apply(lambda x: int(x.replace(".pth", "").split("_")[-1]))
    df = df.drop(axis=1, columns=["Unnamed: 0", "embedder"])
    wandb.log({"table": wandb.Table(dataframe=df)})
    df = df.groupby("epoch")["metric"].mean().reset_index()
    for ep in df.epoch:
        wandb.log({"eval_perfs": df[df.epoch == ep].metric.values[0], "epoch": ep})
    wandb.finish()
