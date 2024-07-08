import os


def get_model_path(MODEL_PATH="backbone_pretrained_models"):
    MODELS = {}
    # For every directory in the folder
    for model_name in os.listdir(MODEL_PATH):
        # For every file in the directory
        if os.path.isdir(os.path.join(MODEL_PATH, model_name)):
            for file_name in os.listdir(os.path.join(MODEL_PATH, model_name)):
                # If the file is a .pth file
                if file_name.endswith(".pth"):
                    MODELS[model_name] = os.path.join(MODEL_PATH, model_name, file_name)
    MODELS[
        "ThreeDInfomax"
    ] = "molDistill/baselines/external_repo/threeDInfomax/configs_clean/tune_QM9_homo.yml"
    MODELS["Not-trained"] = ""
    return MODELS
