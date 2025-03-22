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


def main(args):

    db_path = Path("../data/mol_lmdb")
    if db_path.exists():
        shutil.rmtree(db_path)
    db_path.mkdir(parents=True, exist_ok=True)
    
    db = lmdb.open(
        os.path.join(str(db_path),"1.lmdb"),
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    print(args.mol_dir)
    mol_dir = Path(args.mol_dir)
    path_list = [path for path in mol_dir.glob("*.mol")]

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

    for idx , path in tqdm(enumerate(path_list),total=len(path_list)):
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
        
        # pbar.update(1)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--Tm",
        help="An attributes to be predicted",
        default=True
    )
    parser.add_argument(
        "--Tb",
        help="An attributes to be predicted",
        default=True
    )
    parser.add_argument(
        "--density",
        help="An attributes to be predicted",
        default=True
    )
    parser.add_argument(
        "--flash_point",
        help="An attributes to be predicted",
        default=True
    )
    parser.add_argument(
        "--NHOC",
        help="An attributes to be predicted",
        default=True
    )
    parser.add_argument(
        "--Isp",
        help="An attributes to be predicted",
        default=True
    )

    parser.add_argument(
        "--max_neigh",
        default=50,
    )
    parser.add_argument(
        "--radius",
        default=6,
    )
    parser.add_argument(
        "--mol_dir",
        default="../data/3d_mol"
    )
    parser.add_argument(
        "--get_edges",
        default=False
    )
   
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)