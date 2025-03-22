import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from openbabel import pybel
import os
from wfl.generate import smiles
from wfl.configset import ConfigSet
RANDOM_SEED = 51


coordinate_list = []


def gemerate3D_rdkit(file_path):
    df = pd.read_csv(file_path)[["smile"]]

    for num, row in df.iterrows():
        try:
            smi = row.iloc[0]
            mol = Chem.MolFromSmiles(smi)
            mol = AllChem.AddHs(mol)
            res = AllChem.EmbedMolecule(mol, randomSeed=RANDOM_SEED)
            AllChem.MMFFOptimizeMolecule(mol)
            coordinates = mol.GetConformer().GetPositions()
            heavy_atom_indices = [i for i in range(mol.GetNumAtoms()) if mol.GetAtomWithIdx(i).GetAtomicNum() != 1]
            heavy_atom_coordinates = coordinates[heavy_atom_indices]

            coordinate_list.append(coordinates.astype(np.float32))
            if num % 100 == 0:
                Chem.MolToMolFile(mol, f"{num}.mol")
            coordinate_list.append(coordinates.astype(np.float32))

        except Exception as e:
            print(e)
            print(num, smi)

def generate3D_openbabel(file_path):
    df = pd.read_csv(file_path)[["smile"]]

    for num, row in df.iterrows():
        try:
            smi = row.iloc[0]
            mol = pybel.readstring("smi", smi)
            mol.addh()
            mol.make3D()

            ff = pybel._forcefields["mmff94"]
            ff.Setup(mol.OBMol) 
            ff.SystematicRotorSearch(1000)

            mol.removeh()
            file_name = f"molecule_{smi}.mol"
            output_file = os.path.join("/root/autodl-tmp/myproject/data/3d_mol", file_name)
            mol.write("mol", output_file, overwrite=True)
        except Exception as e:
            print(e)
            print(num, smi)

def generate3d_ase():
    from ase.io import read, write

    mol = read("/root/autodl-tmp/myproject/data/3d_mol/molecule_C1C2C1C1C2C2C3CC3C12.mol")
    print(mol)
    print(mol.get_positions())

if __name__ == "__main__":
    file_path = "/root/autodl-tmp/myproject/data/1006.csv"
    
    generate3d_ase()

