from src import model
from src.data.data_module import MolDataModule
import yaml
from src.model.equiformer_module import EquiformerModule
import argparse
from src.utils.config import build_config


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yml", default="/data/ljp/Project/Protein/equiformer/equiformer_git/src/config/equiformer.yml")

    return parser


def main(config):
    datamodule = MolDataModule(config=config)
    loader = datamodule.train_dataloader() or datamodule.val_dataloader() or datamodule.test_dataloader()

    bond_feat_dim = config["model_attributes"].get("num_gaussians", 50)
    num_targets = 1

    net = model.EquiformerV2_OC20(
        loader.dataset[0].x.shape[-1] if loader and hasattr(loader.dataset[0], "x") and loader.dataset[0].x is not None else None,
        bond_feat_dim,
        num_targets,
        **config["model_attributes"]
    )

    net_module = EquiformerModule(config=config, net=net)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    config = build_config(args)
    main(config)
