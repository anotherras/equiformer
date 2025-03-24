from lightning import LightningModule
import torch.optim as optim
import torch.nn as nn


class EquiformerModule(LightningModule):
    def __init__(self, config, net):
        super().__init__()
        self.config = config

    def forward(self, x):
        pass

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
        return nn.L1Loss()

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
        return {
            "optimizer": optimizer,
            # "lr_scheduler": {
            #     "scheduler": scheduler,
            #     "monitor": "val_loss",
            #     "interval": "epoch",
            #     "frequency": 1,
            # },
        }
