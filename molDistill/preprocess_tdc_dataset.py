import argparse
import json
import os

import datamol as dm
import pandas as pd
import selfies as sf
from tqdm import tqdm

from molDistill.baselines.utils.descriptors import can_be_2d_input
from molDistill.baselines.utils.tdc_dataset import get_dataset
from molDistill.utils.preprocessing import precompute_3d

parser = argparse.ArgumentParser(
    description="Compute ",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--datasets",
    type=str,
    nargs="+",
    default=[
        "ToxCast",
    ],
)

parser.add_argument(
    "--data-path",
    type=str,
    default="../data",
    required=False,
    help="Path to the data folder",
)


def main():
    args = parser.parse_args()
    for dataset in args.datasets:
        dataset = dataset.replace(".csv", "")
        data_path = os.path.join(args.data_path, dataset)

        if not os.path.exists(f"{data_path}/preprocessed.sdf"):
            filepath = f"{data_path}/{dataset}_3d.sdf"
            if os.path.exists(filepath):
                print(f"Loading 3D conformers from data/{dataset}_3d.sdf")
                mols, smiles = precompute_3d(None, filepath)
            else:
                df = get_dataset(dataset.replace("__", " "))
                if "Drug" in df.columns:
                    smiles = df["Drug"].tolist()
                else:
                    smiles = df["smiles"].tolist()
                mols = None
                mols, smiles = precompute_3d(smiles, filepath)

            valid_smiles = []
            valid_mols = []
            for i, s in enumerate(tqdm(smiles, desc="Generating graphs")):
                mol = mols[i]
                # compute molecular weight and limit it under 1000
                desc = dm.descriptors.compute_many_descriptors(mol)
                if desc["mw"] > 1000:
                    continue
                try:
                    _ = sf.encoder(s)
                    if can_be_2d_input(s, mols[i]) and not "." in s:
                        valid_smiles.append(s)
                        valid_mols.append(mols[i])
                except Exception as e:
                    print(f"Error processing {s}: {e}")
                    continue

            smiles = valid_smiles
            mols = valid_mols
            if not os.path.exists(f"{data_path}"):
                os.makedirs(f"{data_path}")

            pre_processed = pd.DataFrame({"smiles": smiles, "mols": mols})
            dm.to_sdf(pre_processed, f"{data_path}/preprocessed.sdf", mol_column="mols")
            # save the SMILES in a json file
            with open(f"{data_path}/smiles.json", "w") as f:
                json.dump(smiles, f)

        else:
            pre_processed = dm.read_sdf(
                f"{data_path}/preprocessed.sdf", as_df=True, mol_column="mols"
            )
            smiles = pre_processed["smiles"].iloc[:, 0].tolist()
            mols = pre_processed["mols"].tolist()



if __name__ == "__main__":
    main()
