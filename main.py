from src.model.equiformer_module import EquiformerModule
from src.data.data_module import MolDataModule
import yaml
from src.model.equiformer_module import EquiformerModule
import argparse
from src.utils.config import build_config
from equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20

import lightning as L
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import datetime


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yml", default="./src/config/equiformer.yml")
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
    parser.add_argument("--seed", default=42, type=int, help="Seed for torch, cuda, numpy")
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
    return parser


def main(config):
    current_time = datetime.datetime.now()
    name = current_time.strftime("%m%d%H%M%S")

    L.seed_everything(config["seed"])

    wandb_logger = None
    # wandb_logger = WandbLogger(
    #     project="mol_equiformer",
    #     name=f"{name}-now",
    #     save_dir="../data/Log",
    # )

    model = config["model"]
    config2 = {
        "name": model.pop("name"),
        "model_attributes": model,
        "dataset": config["dataset"],
        "optim": config["optim"],
        "target_property": config["target_property"],
        "normalize_mean": config["normalize_mean"],
        "normalize_std": config["normalize_std"],
    }

    datamodule = MolDataModule(config=config2)

    bond_feat_dim = config2["model_attributes"].get("num_gaussians", 50)
    bond_feat_dim = 50
    num_targets = 1

    equiformer = EquiformerV2_OC20(None, bond_feat_dim, num_targets, **config2["model_attributes"])

    net_module = EquiformerModule(config=config2, model=equiformer)

    earlystop = EarlyStopping(monitor="val_loss", patience=10, mode="min", verbose=True)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer = Trainer(
        min_epochs=10,
        max_epochs=100,
        accelerator="gpu",
        devices=1,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        callbacks=[lr_monitor],
        enable_progress_bar=True,
        logger=wandb_logger,
        default_root_dir="../data/Log",
        gradient_clip_val=100,
    )
    trainer.fit(model=net_module, datamodule=datamodule)

    if config2["dataset"].get("test", None):
        trainer.test(model=net_module, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    config = build_config(args)
    main(config)
