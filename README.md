This repository contains the code molecular experiments of the paper: "How to distill task-agnostic representations from many teachers?".




# :alembic: Distill representations

To train a model using our distillation framework, we first need to dump the embeddings of the teachers (details on the data preparation can be found here: [Data preparation](#pager-data-processing).

For the teachers used in our project, the embeddings will be automatically computed and saved upon training.
For different teachers, the embeddings should be computed and dumped by the user so that in the data folder, the file: "<model_name>.npy" ("<model_name>_<file_index>.npy" for multi-files) exists.

The training procedure can be launched using the [train_gm.py](molDistill/train_gm.py) script:
```bash
python molDistill/train_gm.py \
  --dataset <dataset_name> \
  --data-path <path_to_dataset> \
  --num-epochs <num_epochs> \
  --embedders-to-simulate <list_of_teachers> \
  --gnn-type <gnn_type> \
  --knifes-config <knifes_config> \
  ...
```
For a complete list of arguments, please refer to the [train_gm.py](molDistill/train_gm.py) script.
The "knifes-config" argument should be a path to a yaml file containing the arguments of the KNIFEs estimators (see [knifes.yaml](hp/knifes.yaml) for an example).

Similarly, L2 and Cosine distillations can be performed using the [train_l2.py](molDistill/train_l2.py) and [train_cos.py](molDistill/train_cos.py) scripts, respectively.

# :framed_picture: Paper's figures

The results of each model on the different downstream tasks are available in the [downstream_eval](downstream_eval) folder.
All figures found in the paper can be re-obtained using the notebooks in the [molDistill/notebooks](molDistill/notebooks) folder.

To evaluate a new model, add the path of the results of the downstream evaluation to the 'MODELS_TO_EVAL' variable in the 'get_all_results' function.


# :pager: Data processing

All datasets were pre-processed following the same procedure.
Two options are available to process the data, depending on the size of the dataset:
- For small datasets, the data can be processed using the [process_tdc_dataset.py](molDistill/preprocess_tdc_dataset.py) script:
```bash
python molDistill/process_tdc_data.py --dataset <dataset_name> --data-path <path_to_dataset>
```
- For large datasets, the data can be processed using the [process_tdc_dataset_multifiles.py](molDistill/preprocess_tdc_dataset_multifiles.py) script:
```bash
python molDistill/process_tdc_data_multifiles.py --dataset <dataset_name> --data-path <path_to_dataset> --i0 <initial_index_to_process> --step <datapoints_per_files>
```
Using the second scripts, the dataset will be split into multiple files, each containing 'step' datapoints.

