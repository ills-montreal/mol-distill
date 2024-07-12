import os

import datamol as dm
from tqdm import tqdm

from molDistill.baselines.utils.descriptors import can_be_2d_input


def compute_3d(smiles: str):
    if "." in smiles:
        return None
    mol = dm.to_mol(smiles)
    if not can_be_2d_input(smiles, mol):
        return None
    desc = dm.descriptors.compute_many_descriptors(mol)
    if desc["mw"] > 1000:
        return None

    try:
        mol = dm.conformers.generate(
            dm.to_mol(smiles),
            align_conformers=True,
            ignore_failure=True,
            num_threads=8,
            n_confs=5,
        )
        return mol
    except Exception as e:
        print(e)
        return None


def precompute_3d(
    smiles,
    filepath,
):
    if os.path.exists(filepath):
        mols = dm.read_sdf(filepath)
        smiles = [dm.to_smiles(m, True, False) for m in mols]
        return mols, smiles

    mols = []
    for s in tqdm(smiles):
        mol = compute_3d(s)
        if mol is not None:
            mols.append(mol)

    smiles = [dm.to_smiles(m, True, False) for m in mols]
    dm.to_sdf(mols, filepath)

    return mols, smiles
