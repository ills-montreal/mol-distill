import os
import logging
import time

import argparse

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# while true; launch the downstream_eval.py script if a new model is available in the folder specified
# by the MODEL_PATH variable that has not been seen before

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("MODEL_PATH", type=str)
    args = parser.parse_args()
    MODEL_PATH = args.MODEL_PATH
    checked_models = ["best_model.pth"]
    analyze = True
    while True:
        # Get all the models in the folder
        models = os.listdir(MODEL_PATH)
        # For every model
        for model in models:
            # If the model has not been checked
            if model.endswith(".pth") and not model in checked_models:
                # Launch the downstream eval script
                os.system(
                    f"python molDistill/downstream_eval.py --embedders custom:{os.path.join(MODEL_PATH, model)}"
                )
                # Add the model to the checked models
                checked_models.append(model)

            # if stop.txt is in the directory, stop the script
            if model == "stop.txt":
                if not analyze:
                    break
                analyze = False

        logger.info("Sleeping for 60 seconds")
        time.sleep(10)
