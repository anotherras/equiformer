from src.utils.atom2graph import AtomsToGraphs
import argparse
from pathlib import Path
from tqdm import tqdm
from ase.io import read, write
import torch
import lmdb
import pickle
import os
import shutil
import random
from sklearn.model_selection import train_test_split
import pandas as pd


def create_db(db_path, path_list):
    db_path = Path(db_path)
    if db_path.exists():
        shutil.rmtree(db_path)
    db_path.mkdir(parents=True, exist_ok=True)

    db = lmdb.open(
        os.path.join(str(db_path), "mol.lmdb"),
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    a2g = AtomsToGraphs(
        max_neigh=args.max_neigh,
        radius=args.radius,
        r_Tm=args.Tm,
        r_Tb=args.Tb,
        r_density=args.density,
        r_flash_point=args.flash_point,
        r_NHOC=args.NHOC,
        r_Isp=args.Isp,
        r_fixed=True,
        r_distances=False,
        r_edges=args.get_edges,
    )

    for idx, path in tqdm(enumerate(path_list), total=len(path_list)):
        atoms = read(path)
        sid = path.stem.split("_")[-1]
        atoms.molecule_name = sid
        data_object = a2g.convert(atoms)

        data_object.tags = torch.LongTensor(atoms.get_tags())
        data_object.sid = sid

        txn = db.begin(write=True)
        txn.put(
            f"{idx}".encode("ascii"),
            pickle.dumps(data_object, protocol=-1),
        )
        txn.commit()
        idx += 1

    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(idx, protocol=-1))
    txn.commit()

    db.sync()
    db.close()


def main(args):

    mol_dir = Path(args.mol_dir)
    data = [path for path in mol_dir.glob("*.mol")]

    train_data, temp_data = train_test_split(data, train_size=0.8, random_state=42)
    val_ratio_adjusted = 0.1 / (1 - 0.8)
    val_data, test_data = train_test_split(temp_data, train_size=val_ratio_adjusted, random_state=42)

    db_list = ["../data/train_lmdb", "../data/val_lmdb", "../data/test_lmdb"]
    datapath_list = [train_data, val_data, test_data]

    mean_path = "../data/mean.csv"
    std_path = "../data/std.csv"
    if not (os.path.exists(mean_path) and os.path.exists(std_path)):
        train_mol = [i.stem.split("_")[-1] for i in train_data]
        raw_data = pd.read_csv("../data/1006.csv", index_col=0).loc[train_mol]
        mean = raw_data.mean()
        std = raw_data.std()

        mean.to_csv(mean_path)
        std.to_csv(std_path)
    else:
        pass

    for db_path, data_path in zip(db_list, datapath_list):
        create_db(db_path, data_path)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--Tm", help="An attributes to be predicted", default=True)
    parser.add_argument("--Tb", help="An attributes to be predicted", default=True)
    parser.add_argument("--density", help="An attributes to be predicted", default=True)
    parser.add_argument("--flash_point", help="An attributes to be predicted", default=True)
    parser.add_argument("--NHOC", help="An attributes to be predicted", default=True)
    parser.add_argument("--Isp", help="An attributes to be predicted", default=True)

    parser.add_argument(
        "--max_neigh",
        default=50,
    )
    parser.add_argument(
        "--radius",
        default=6,
    )
    parser.add_argument("--mol_dir", default="../data/3d_mol")
    parser.add_argument("--get_edges", default=False)

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
