from src.utils.atom2graph import AtomsToGraphs
import argparse


def main(args):
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

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--Tm",
        help="An attributes to be predicted",
    )
    parser.add_argument(
        "--Tb",
        help="An attributes to be predicted",
    )
    parser.add_argument(
        "--density",
        help="An attributes to be predicted",
    )
    parser.add_argument(
        "--flash_point",
        help="An attributes to be predicted",
    )
    parser.add_argument(
        "--NHOC",
        help="An attributes to be predicted",
    )
    parser.add_argument(
        "--Isp",
        help="An attributes to be predicted",
    )

    parser.add_argument(
        "--max_neigh",
        default=50,
    )
    parser.add_argument(
        "--radius",
        default=6,
    )
   
    return parser


if __name__ == "__main__":
    pass