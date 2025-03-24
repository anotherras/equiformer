from copy import copy
from pathlib import Path
import yaml


def build_config(args):
    config = yaml.safe_load(open(args.config_yml, "r"))

    # # Some other flags.
    # config["mode"] = args.mode
    # config["identifier"] = args.identifier
    # config["timestamp_id"] = args.timestamp_id
    # config["seed"] = args.seed
    # config["is_debug"] = args.debug
    # config["run_dir"] = args.run_dir
    # config["print_every"] = args.print_every
    # config["amp"] = args.amp
    # config["checkpoint"] = args.checkpoint
    # config["cpu"] = args.cpu
    # # Submit
    # config["submit"] = args.submit
    # config["summit"] = args.summit
    # # Distributed
    # config["local_rank"] = args.local_rank
    # config["distributed_port"] = args.distributed_port
    # config["world_size"] = args.num_nodes * args.num_gpus
    # config["distributed_backend"] = args.distributed_backend
    # config["noddp"] = args.no_ddp
    # config["gp_gpus"] = args.gp_gpus

    return config
