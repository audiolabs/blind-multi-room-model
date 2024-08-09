from typing import Any
from collections import deque
import pickle
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric

from src.utils.signals import DFNInference


class MultiSlopeEstimation(LightningModule):
    """Implements the lightning module for downstream training and evaluation"""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        component: nn.Module,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.component = component

        # init metrics
        self.val_loss_best = MinMetric()
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.eval_output = deque()

        # load DecayFitNet inference wrapper
        self.dfi = DFNInference()
        # store EDC normalization factor
        self.nfactor = self.dfi.input_transform["edcs_db_normfactor"]

        # time axis, hard-coded
        self.time_axis = torch.linspace(0, 1.96162498, 100)

    def forward(self, x: Tensor) -> Tensor:
        return self.component(x)

    def on_train_start(self):
        self.val_loss.reset()
        self.val_loss_best.reset()
        print("Training Start")

    def step(self, batch: Any) -> Tensor:
        # unpack
        signal, edcs, pos, ls_id, scene_id, t60, edc_dfn = batch

        y_hat = self.forward(signal.unsqueeze(1))

        loss = F.mse_loss(y_hat, t60)

        step_dict = {
            "loss": loss,
            "estimate": y_hat,
            "label": edcs,
            "ls_id": ls_id,
            "scene_id": scene_id,
            "t60": t60,
        }

        return step_dict

    def training_step(self, batch: Any, batch_idx: int) -> dict:
        step_dict = self.step(batch)

        self.train_loss(step_dict["loss"])
        self.log(
            f"train/loss",
            step_dict["loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return step_dict

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int) -> dict:
        step_dict = self.step(batch)
        self.val_loss(step_dict["loss"])
        self.log(
            f"val/loss", step_dict["loss"], on_step=False, on_epoch=True, prog_bar=True
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        step_dict = self.step(batch)

        self.log(
            f"test/loss", step_dict["loss"], on_step=False, on_epoch=True, prog_bar=True
        )

        self.eval_output.append(
            {
                "loss": step_dict["loss"].cpu(),
                "estimate": step_dict["estimate"].cpu(),
                "label": step_dict["label"].cpu(),
                "ls_id": step_dict["ls_id"].cpu(),
                "scene_id": step_dict["scene_id"].cpu(),
                "t60": step_dict["t60"].cpu(),
            }
        )

    def on_test_epoch_end(self):
        output_file = f"{self.logger.log_dir}/outputs.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(self.eval_output, f)

        # store model state
        self.component.cpu()
        states = {"component_state": self.component.state_dict()}
        states_file = f"{self.logger.log_dir}/states.pkl"
        self.upstream_state = states_file
        with open(states_file, "wb") as f:
            pickle.dump(states, f)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
