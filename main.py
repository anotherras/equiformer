import copy
import logging
import os
import sys
import time
from pathlib import Path

from src.ocpmodels.common import distutils
from src.ocpmodels.common.flags import flags
from src.ocpmodels.common.registry import registry
from src.ocpmodels.common.utils import (
    build_config,
    create_grid,
    save_experiment_log,
    setup_imports,
    setup_logging,
)

from src import model
from src import trainer
import src.trainer.dist_setup


from src.data.data_module import MolDataModule
import yaml
from src.model.equiformer_module import EquiformerModule


def main(config_path):
    config = yaml.safe_load(open(config_path, "r"))

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

    main()

    # setup_logging()

    # parser = flags.get_parser()
    # args, override_args = parser.parse_known_args()
    # config = build_config(args, override_args)

    # if args.submit:  # Run on cluster
    #     slurm_add_params = config.get(
    #         "slurm", None
    #     )  # additional slurm arguments
    #     if args.sweep_yml:  # Run grid search
    #         configs = create_grid(config, args.sweep_yml)
    #     else:
    #         configs = [config]

    #     logging.info(f"Submitting {len(configs)} jobs")
    #     executor = submitit.AutoExecutor(
    #         folder=args.logdir / "%j", slurm_max_num_timeout=3
    #     )
    #     executor.update_parameters(
    #         name=args.identifier,
    #         mem_gb=args.slurm_mem,
    #         timeout_min=args.slurm_timeout * 60,
    #         slurm_partition=args.slurm_partition,
    #         gpus_per_node=args.num_gpus,
    #         cpus_per_task=(config["optim"]["num_workers"] + 1),
    #         tasks_per_node=(args.num_gpus if args.distributed else 1),
    #         nodes=args.num_nodes,
    #         slurm_additional_parameters=slurm_add_params,
    #     )
    #     for config in configs:
    #         config["slurm"] = copy.deepcopy(executor.parameters)
    #         config["slurm"]["folder"] = str(executor.folder)
    #     jobs = executor.map_array(Runner(), configs)
    #     logging.info(
    #         f"Submitted jobs: {', '.join([job.job_id for job in jobs])}"
    #     )
    #     log_file = save_experiment_log(args, jobs, configs)
    #     logging.info(f"Experiment log saved to: {log_file}")

    # else:  # Run locally
    #     Runner()(config)
