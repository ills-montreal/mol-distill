from typing import Dict, List
import numpy as np

import datamol as dm
import pandas as pd
from tdc.generation import MolGen
from tdc.multi_pred import DTI
from tdc.single_pred import Tox, ADME, HTS, QM
from tdc.utils import retrieve_label_name_list
from tqdm import tqdm

# Correspondancy between dataset name and the corresponding prediction/generation TDC problem
correspondancy_dict = {
    "Tox21": Tox,
    "ToxCast": Tox,
    "LD50_Zhu": Tox,
    "hERG": Tox,
    "herg_central": Tox,
    "hERG_Karim": Tox,
    "AMES": Tox,
    "DILI": Tox,
    "Skin__Reaction": Tox,
    "Skin Reaction": Tox,
    "Carcinogens_Lagunin": Tox,
    "ClinTox": Tox,
    "Caco2_Wang": ADME,
    "PAMPA_NCATS": ADME,
    "HIA_Hou": ADME,
    "Pgp_Broccatelli": ADME,
    "Bioavailability_Ma": ADME,
    "Lipophilicity_AstraZeneca": ADME,
    "Solubility_AqSolDB": ADME,
    "HydrationFreeEnergy_FreeSolv": ADME,
    "BBB_Martins": ADME,
    "PPBR_AZ": ADME,
    "VDss_Lombardo": ADME,
    "CYP2C19_Veith": ADME,
    "CYP2D6_Veith": ADME,
    "CYP3A4_Veith": ADME,
    "CYP1A2_Veith": ADME,
    "CYP2C9_Veith": ADME,
    "CYP2C9_Substrate_CarbonMangels": ADME,
    "CYP2D6_Substrate_CarbonMangels": ADME,
    "CYP3A4_Substrate_CarbonMangels": ADME,
    "Half_Life_Obach": ADME,
    "Clearance_Hepatocyte_AZ": ADME,
    "Clearance_Microsome_AZ": ADME,
    "SARSCoV2_Vitro_Touret": HTS,
    "SARSCoV2_3CLPro_Diamond": HTS,
    "HIV": HTS,
    "orexin1_receptor_butkiewicz": HTS,
    "m1_muscarinic_receptor_agonists_butkiewicz": HTS,
    "m1_muscarinic_receptor_antagonists_butkiewicz": HTS,
    "potassium_ion_channel_kir2.1_butkiewicz": HTS,
    "kcnq2_potassium_channel_butkiewicz": HTS,
    "cav3_t-type_calcium_channels_butkiewicz": HTS,
    "choline_transporter_butkiewicz": HTS,
    "serine_threonine_kinase_33_butkiewicz": HTS,
    "tyrosyl-dna_phosphodiesterase_butkiewicz": HTS,
    "QM7b": QM,
    "QM8": QM,
    "QM9": QM,
    "MOSES": MolGen,
    "ZINC": MolGen,
    "ChEMBL": MolGen,
    "DAVIS": DTI,
    "BindingDB_Kd": DTI,
    "BindingDB_IC50": DTI,
    "KIBA": DTI,
    "BindingDB_Ki": DTI,
}

correspondancy_dict_DTI = {
    "DAVIS": DTI,
    "BindingDB_Kd": DTI,
    "BindingDB_IC50": DTI,
    "KIBA": DTI,
    "BindingDB_Ki": DTI,
}

THRESHOLD_MIN_SAMPLES = 128


def preprocess_smiles(s):
    mol = dm.to_mol(s)
    return dm.to_smiles(mol, True, False)


class EvaluationDatasetIterable:
    def __init__(
        self,
        dataset: str,
        smiles: List[str],
        random_seed: int = 42,
        method: str = "random",
    ):
        self.dataset = dataset
        try:
            self.label_list = retrieve_label_name_list(dataset)
        except Exception as e:
            self.label_list = [None]
        self.random_seed = random_seed
        self.method = method
        self.smiles = smiles
        self.smiles_to_idx = {s: i for i, s in enumerate(self.smiles)}

    def sample(self):
        for label in self.label_list:
            tdc_dataset = correspondancy_dict[self.dataset](
                name=self.dataset, label_name=label
            )
            non_valid = True
            i_rs = 0
            while non_valid:
                split = tdc_dataset.get_split(
                    seed=self.random_seed + i_rs * 100, method=self.method
                )
                non_valid = False
                for k in split:
                    non_valid = non_valid or split[k].Y.nunique() < 2
                i_rs += 1
                if not non_valid:
                    split_idx = self.get_split_idx(split)

                    for k in split_idx.keys():
                        non_valid = non_valid or len(np.unique(split_idx[k]["y"])) < 2
            yield split_idx

    def get_split_idx(self, split: Dict[str, pd.DataFrame]):
        for key in split.keys():
            split[key]["prepro_smiles"] = dm.parallelized(
                preprocess_smiles, split[key]["Drug"].tolist()
            )

        split_idx = {}
        for key in split.keys():
            split_idx[key] = {"x": [], "y": []}
            for smile, y in zip(split[key]["prepro_smiles"], split[key]["Y"]):
                if smile in self.smiles:
                    split_idx[key]["x"].append(self.smiles_to_idx[smile])
                    split_idx[key]["y"].append(y)

        return split_idx


def get_dataset(dataset: str):
    try:
        data = correspondancy_dict[dataset](name=dataset)
        if dataset in correspondancy_dict_DTI.keys():
            data.convert_to_log(form="binding")
            df = data.harmonize_affinities(mode="max_affinity")
        else:
            df = data.get_data()

    except Exception as e:
        if e.args[0].startswith(
            "Please select a label name. You can use tdc.utils.retrieve_label_name_list"
        ):
            label_list = retrieve_label_name_list(dataset)
            df = []
            for l in tqdm(label_list):
                df.append(
                    correspondancy_dict[dataset](name=dataset, label_name=l).get_data()
                )
            df = pd.concat(df).drop_duplicates("Drug")

    if hasattr(df, "Target_ID"):
        allowed_targets = (
            df["Target_ID"]
            .value_counts()[df["Target_ID"].value_counts() > THRESHOLD_MIN_SAMPLES]
            .index
        )
        df = df[df["Target_ID"].isin(allowed_targets)]
    return df


def get_dataset_split_DTI(dataset, random_seed=42):
    data = correspondancy_dict_DTI[dataset](name=dataset)
    data.convert_to_log(form="binding")
    data = data.harmonize_affinities(mode="max_affinity")
    split = []
    for id in data["Target_ID"].unique():
        subdata = data[data["Target_ID"] == id][["Drug", "Target_ID", "Y"]]
        if subdata.shape[0] < THRESHOLD_MIN_SAMPLES:
            continue
        print(subdata.shape)
        split.append(
            {"train": data[data["Target_ID"] == id][["Drug", "Target_ID", "Y"]]}
        )
    return split
