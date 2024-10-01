# mol-distill

This repository contains the code molecular experiments of the paper: "How to distill task-agnostic representations from many teachers?".




## :framed_picture: Paper's figures

The results of each model on the different downstream tasks are available in the [downstream_eval](downstream_eval) folder.
All figures found in the paper can be re-obtained using the notebooks in the [molDistill/notebooks](molDistill/notebooks) folder.

To evaluate a new model, add the path of the results of the downstream evaluation to the 'MODELS_TO_EVAL' variable in the 'get_all_results' function.
