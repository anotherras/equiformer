from src.model.equiformer_module import EquiformerModule
from src.data.data_module import MolDataModule
import yaml
from src.model.equiformer_module import EquiformerModule
import argparse
from src.utils.config import build_config
from equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20

import lightning as L
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import datetime


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yml", default="/data/ljp/Project/Protein/equiformer/equiformer_git/src/config/equiformer.yml")
    parser.add_argument(
        "--mode",
        choices=["train", "predict", "run-relaxations", "validate"],
        default="train",
        help="Whether to train the model, make predictions, or to run relaxations",
    )
    parser.add_argument(
        "--identifier",
        default="",
        type=str,
        help="Experiment identifier to append to checkpoint/log/result directory",
    )
    parser.add_argument("--seed", default=0, type=int, help="Seed for torch, cuda, numpy")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether this is a debugging run or not",
    )
    parser.add_argument(
        "--run-dir",
        default="./",
        type=str,
        help="Directory to store checkpoint/log/result directory",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
    )
    return parser


def main(config):
    current_time = datetime.datetime.now()
    name = current_time.current_time.strftime("%m%d%H%M%S")

    L.seed_everything(config.seed)

    wandb_logger = WandbLogger(
        project="mol_equiformer",
        name=f"{name}-now",
        save_dir="../data/Log",
    )

    datamodule = MolDataModule(config=config)

    model = config["model"]
    config2 = {"name": model.pop("name"), "model_attributes": model}

    bond_feat_dim = config2["model_attributes"].get("num_gaussians", 50)
    bond_feat_dim = 50
    num_targets = 1

    net = EquiformerV2_OC20(None, bond_feat_dim, num_targets, **config2["model_attributes"])

    net_module = EquiformerModule(config=config, net=net)

    earlystop = EarlyStopping(monitor="val_loss", patience=10, mode="min", verbose=True)
    trainer = Trainer(
        min_epochs=10,
        max_epochs=10000,
        accelerator="gpu",
        devices=1,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        callbacks=[earlystop],
        enable_progress_bar=True,
        logger=wandb_logger,
        default_root_dir="/data/ljp/Project/Polymer/sol_polygnn/pl/logs",
    )
    trainer.fit(model=net_module, datamodule=datamodule)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    config = build_config(args)
    main(config)
