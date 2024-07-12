import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

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
        "MOSES",
    ],
)

parser.add_argument(
    "--data-path",
    type=str,
    default="../data",
    required=False,
    help="Path to the data folder",
)

parser.add_argument("--i0", type=int, default=0, help="Starting index for the dataset")

parser.add_argument(
    "--step", type=int, default=100000, help="Step size for the dataset"
)


def main():
    args = parser.parse_args()
    i0 = args.i0
    step = args.step

    for dataset in args.datasets:
        dataset = dataset.replace(".csv", "")
        data_path = os.path.join(args.data_path, dataset)

        preprocessed_dir = f"{data_path}/preprocessed"
        td_dir = f"{data_path}/3d"

        os.makedirs(preprocessed_dir, exist_ok=True)
        os.makedirs(td_dir, exist_ok=True)

        preprocessed_filepath = f"{preprocessed_dir}/preprocessed_{i0}.sdf"

        if not os.path.exists(preprocessed_filepath):
            filepath = os.path.join(td_dir, f"{dataset}_{i0}_3d.sdf")
            if os.path.exists(filepath):
                print(f"Loading 3D conformers from data/{dataset}_3d.sdf")
                mols, smiles = precompute_3d(None, filepath)
            else:
                df = get_dataset(dataset.replace("__", " ")).iloc[
                    i0 * step : min(i0 * step + step, len(df))
                ]
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
                    if not "." in s and can_be_2d_input(s, mols[i]):
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
            dm.to_sdf(pre_processed, preprocessed_filepath, mol_column="mols")
            # save the SMILES in a json file

        else:
            pre_processed = dm.read_sdf(
                f"{data_path}/preprocessed.sdf", as_df=True, mol_column="mols"
            )
            smiles = pre_processed["smiles"].iloc[:, 0].tolist()
            mols = pre_processed["mols"].tolist()


if __name__ == "__main__":
    main()
