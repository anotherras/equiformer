from lightning import LightningModule
import torch.optim as optim
import torch.nn as nn
from ocpmodels.common.data_parallel import OCPDataParallel
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.functional import mse_loss
from torcheval.metrics.functional import r2_score, mean_squared_error
from torchmetrics.functional.regression import mean_absolute_error
import pandas as pd
import math
from torch.optim.lr_scheduler import LambdaLR

# from trainer.lr_scheduler import LRScheduler


class EquiformerModule(LightningModule):
    def __init__(self, config, model, normalizer=None):
        super().__init__()
        self.config = config
        self.model = model
        self.target_porperty = self.config["target_property"]
        self.loss = self.load_loss()

        if hasattr(self.model, "no_weight_decay"):
            self.model_params_no_wd = self.model.no_weight_decay()

        self.model = OCPDataParallel(
            self.model,
            output_device="cuda",
            num_gpus=1,
        )

        if self.config.get("dataset", None) is not None and normalizer is None:
            self.normalizer = self.config["dataset"]
            self.normalizers = self.load_normalizer(self.normalizer)

    def forward(self, batch):
        out = self.model(batch)
        return out

    def computer_loss(self, out, batch):

        target = torch.cat([getattr(b, self.target_porperty).to(self.device) for b in batch])
        if self.normalizer.get("normalize_labels", False):
            target = self.normalizers["target"].norm(target)

        # 源代码中有但不知道有什么用,可能是平衡两方面loss
        energy_mult = self.config["optim"].get("energy_coefficient", 1)

        return self.loss(out, target)

    def training_step(self, batch, batch_idx):

        out = self.forward(batch)
        loss = self.computer_loss(out, batch)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(batch))
        return loss

    def _r2_score(self, out, batch):
        if len(out) <= 1:
            return 0
        target = torch.cat([getattr(b, self.target_porperty).to(self.device) for b in batch])
        out_denorm = self.normalizers["target"].denorm(out)
        return r2_score(out_denorm, target)

    def _mae(self, out, batch):
        target = torch.cat([getattr(b, self.target_porperty).to(self.device) for b in batch])
        out_denorm = self.normalizers["target"].denorm(out)
        return mean_absolute_error(out_denorm, target)

    def validation_step(self, batch, batch_idx):

        out = self.forward(batch)
        loss = self.computer_loss(out, batch)

        r2 = self._r2_score(out, batch)
        mae = self._mae(out, batch)

        self.log("val_loss", loss, on_step=False, on_epoch=True, batch_size=len(batch))
        self.log("val_r2", r2, on_step=False, on_epoch=True, batch_size=len(batch))
        self.log("val_mae", mae, on_step=False, on_epoch=True, batch_size=len(batch))

        # self.log("val_rmse", rmse, on_step=False, on_epoch=True, batch_size=len(batch))

    def test_step(self, batch, batch_idx):

        out = self.forward(batch)
        loss = self.computer_loss(out, batch)

        r2 = self._r2_score(out, batch)

        self.log("test_loss", loss, on_step=False, on_epoch=True, batch_size=len(batch))
        self.log("test_r2", r2, on_step=False, on_epoch=True, batch_size=len(batch))

    def load_normalizer(self, normalizer):
        import numpy as np

        mean = pd.read_csv(self.config["normalize_mean"], index_col=0).loc[self.target_porperty].astype(np.float32)
        std = pd.read_csv(self.config["normalize_std"], index_col=0).loc[self.target_porperty].astype(np.float32)

        normalizers = {}

        if normalizer.get("normalize_labels", False):
            normalizers["target"] = Normalizer(mean=mean, std=std, device="cuda")
            # if "target_mean" in normalizer:
            #     normalizers["target"] = Normalizer(
            #         mean=self.normalizer["target_mean"],
            #         std=self.normalizer["target_std"],
            #         device=self.device,
            #     )
            # else:
            #     normalizers["target"] = Normalizer(
            #         tensor=self.train_loader.dataset.data.y[self.train_loader.dataset.__indices__],
            #         device=self.device,
            #     )
        return normalizers

    def add_weight_decay(self, model, weight_decay, skip_list=()):
        decay = []
        no_decay = []
        name_no_wd = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if (
                name.endswith(".bias")
                or name.endswith(".affine_weight")
                or name.endswith(".affine_bias")
                or name.endswith(".mean_shift")
                or "bias." in name
                or any(name.endswith(skip_name) for skip_name in skip_list)
            ):
                no_decay.append(param)
                name_no_wd.append(name)
            else:
                decay.append(param)
        name_no_wd.sort()
        params = [{"params": no_decay, "weight_decay": 0.0}, {"params": decay, "weight_decay": weight_decay}]
        return params, name_no_wd

    def load_loss(self):
        # return nn.L1Loss()
        return nn.MSELoss()

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()

    def configure_optimizers(self):
        optimizer = self.config["optim"].get("optimizer", "AdamW")
        optimizer = getattr(optim, optimizer)
        optimizer_params = self.config["optim"]["optimizer_params"]
        weight_decay = optimizer_params["weight_decay"]

        parameters, name_no_wd = self.add_weight_decay(self.model, weight_decay, self.model_params_no_wd)

        self.optimizer = optimizer(
            parameters,
            lr=self.config["optim"]["lr_initial"],
            **optimizer_params,
        )
        # if ues ReduceLROnPlateau
        # self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=0.1, patience=5, min_lr=1e-6, verbose=True)
        # if use cos LRScheduler like paper
        self.scheduler = self.load_scheduler()
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                # "monitor": "val_loss",
                "interval": "step",
                # "frequency": 1,
            },
        }

    def load_scheduler(self):

        total_steps = self.trainer.estimated_stepping_batches

        scheduler_params = self.config["optim"]["scheduler_params"]
        scheduler_params["epochs"] = total_steps
        scheduler_params["warmup_epochs"] = int(scheduler_params["warmup_epochs"] * total_steps)

        fn = CosineLRLambda(scheduler_params)

        return LambdaLR(self.optimizer, fn)


class Normalizer(object):
    """Normalize a Tensor and restore it later."""

    def __init__(self, tensor=None, mean=None, std=None, device=None):
        """tensor is taken as a sample to calculate the mean and std"""
        if tensor is None and mean is None:
            return

        if device is None:
            device = "cpu"

        if tensor is not None:
            self.mean = torch.mean(tensor, dim=0).to(device)
            self.std = torch.std(tensor, dim=0).to(device)
            return

        if mean is not None and std is not None:
            self.mean = torch.tensor(mean).to(device)
            self.std = torch.tensor(std).to(device)

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"].to(self.mean.device)
        self.std = state_dict["std"].to(self.mean.device)


class CosineLRLambda:
    def __init__(self, scheduler_params):
        self.warmup_epochs = scheduler_params["warmup_epochs"]
        self.lr_warmup_factor = scheduler_params["warmup_factor"]
        self.max_epochs = scheduler_params["epochs"]
        self.lr_min_factor = scheduler_params["lr_min_factor"]

    def __call__(self, current_step):
        # `warmup_epochs` is already multiplied with the num of iterations
        if current_step <= self.warmup_epochs:
            alpha = current_step / float(self.warmup_epochs)
            return self.lr_warmup_factor * (1.0 - alpha) + alpha
        else:
            if current_step >= self.max_epochs:
                return self.lr_min_factor
            lr_scale = self.lr_min_factor + 0.5 * (1 - self.lr_min_factor) * (1 + math.cos(math.pi * (current_step / self.max_epochs)))
            return lr_scale
