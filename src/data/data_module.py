import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from .lmdb_dataset import LMDBDataset
from .data_parallel import BalancedBatchSampler,ParallelCollater


class MolDataModule(LightningDataModule):
    def __init__(self, config,cpu=False):
        super().__init__()
        self.config = config

        if torch.cuda.is_available() and not self.cpu:
            self.device = torch.device("cuda")

        self.train_dataloader = self.val_dataloader = self.test_dataloader = None


    def prepare_data(self):
        pass

    def setup(self, stage):
        self.parallel_collater = ParallelCollater(
            0 if self.cpu else 1,
            self.config["model_attributes"].get("otf_graph", False),
        )
        self.train_dataset = LMDBDataset(self.config["dataset"])
        self.train_sampler = self.get_sampler(
                self.train_dataset,
                self.config["optim"]["batch_size"],
                shuffle=True,
            )
        self.train_loader = self.get_dataloader(
                self.train_dataset,
                self.train_sampler,
            )
        
        if self.config.get("val_dataset", None):
            self.val_dataset = LMDBDataset(self.config["val_dataset"])
            self.val_sampler = self.get_sampler(
                self.val_dataset,
                self.config["optim"].get(
                    "eval_batch_size", self.config["optim"]["batch_size"]
                ),
                shuffle=False,
            )
            self.val_loader = self.get_dataloader(
                self.val_dataset,
                self.val_sampler,
            )

        if self.config.get("test_dataset", None):
            self.test_dataset = LMDBDataset(self.config["test_dataset"])
            self.test_sampler = self.get_sampler(
                self.test_dataset,
                self.config["optim"].get(
                    "eval_batch_size", self.config["optim"]["batch_size"]
                ),
                shuffle=False,
            )
            self.test_loader = self.get_dataloader(
                self.test_dataset,
                self.test_sampler,
            )

    def train_dataloader(self):
        return self.train_dataloader

    def val_dataloader(self):
        return self.val_dataloader if self.val_dataloader else None


    def test_dataloader(self):
        return self.test_dataloader if self.test_dataloader else None   
    
    def get_sampler(self, dataset, batch_size, shuffle):
        if "load_balancing" in self.config["optim"]:
            balancing_mode = self.config["optim"]["load_balancing"]
            force_balancing = True
        else:
            balancing_mode = "atoms"
            force_balancing = False

        sampler = BalancedBatchSampler(
            dataset,
            batch_size=batch_size,
            num_replicas=1,
            rank=0,
            device=self.device,
            mode=balancing_mode,
            shuffle=shuffle,
            force_balancing=force_balancing,
        )
        return sampler

    def get_dataloader(self, dataset, sampler):
        loader = DataLoader(
            dataset,
            collate_fn=self.parallel_collater,
            num_workers=self.config["optim"]["num_workers"],
            pin_memory=True,
            batch_sampler=sampler,
        )
        return loader
    