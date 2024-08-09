from typing import Any
from collections import deque
import pickle
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
from hydra import compose
from hydra.utils import instantiate
from omegaconf import DictConfig
from rich.console import Console
from rich.table import Table

from src.utils.signals import DFNInference


class MultiSlopeEstimation(LightningModule):
    """Implements the lightning module for downstream training and evaluation"""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        mono_weight: float,
        component: nn.Module,
        t60_estimator: DictConfig | None,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.component = component
        self.mono_weight = mono_weight

        # init metrics
        self.val_loss_best = MinMetric()
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.eval_output = deque()

        # init DecayFitNet inference
        self.dfi = DFNInference()

        # store EDC normalization factor
        self.nfactor = self.dfi.input_transform["edcs_db_normfactor"]

        # setup blind t60 estimator
        t60_estimator_cfg = compose(t60_estimator.cfg)
        t60_estimator_cfg.model.component.state = t60_estimator.state
        self.t60_estimator = instantiate(t60_estimator_cfg.model.component)

        # freeze t60 estimator
        for param in self.t60_estimator.parameters():
            param.requires_grad = False

        # time axis, hard-coded, in-line with DecayFitNet parameters
        self.register_buffer("time_axis", torch.linspace(0, 1.96162498, 100))

    def forward(self, x: Tensor) -> Tensor:
        return self.component(x)

    def on_train_start(self):
        self.val_loss.reset()
        self.val_loss_best.reset()
        print("Training Start")

    def step(self, batch: Any) -> Tensor:

        signal, edc, pos, ls_id, scene_id, t60, edc_dfn = batch

        signal = signal.unsqueeze(1)

        edc_hat = self.forward(signal)
        edc_hat = edc_hat.view(edc.shape)

        # monotonicity penalty
        diff = torch.diff(edc_hat, dim=-1)
        diff = torch.mean(diff[diff > 0])

        loss = F.mse_loss(edc_hat, edc) + self.mono_weight * diff

        # scale back to dB (hard-coded range)
        edc_hat = (edc_hat - 1) * 70
        edc = (edc - 1) * 70

        # get blind t60 estimate from baseline
        t60_hat = self.t60_estimator(signal)

        # compute linear edc from blind t60 estimate
        edc_from_t60 = torch.einsum("nf,t->nft", [(-60 / t60_hat), self.time_axis])

        step_dict = {
            "loss": loss,
            "estimate": edc_hat,
            "label": edc,
            # "norm": norm,
            "ls_id": ls_id,
            "scene_id": scene_id,
            "t60": t60,
            "edc_from_t60": edc_from_t60,
            "edc_dfn": edc_dfn,
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
                "edc_from_t60": step_dict["edc_from_t60"].cpu(),
                "edc_dfn": step_dict["edc_dfn"].cpu(),
            }
        )

    def on_test_epoch_end(self):
        output_file = f"{self.logger.log_dir}/outputs.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(self.eval_output, f)

        freqs = [125, 250, 500, 1000, 2000, 4000, 8000]

        estimates = torch.cat([batch["estimate"] for batch in self.eval_output])
        estimates_t60 = torch.cat([batch["edc_from_t60"] for batch in self.eval_output])
        estimates_dfn = torch.cat([batch["edc_dfn"] for batch in self.eval_output])
        labels = torch.cat([batch["label"] for batch in self.eval_output])
        # estimates = rearrange(estimates, "b f t -> f b t")
        # labels = rearrange(labels, "b f t -> f b t")

        maes, maes_t60, maes_dfn = [], [], []
        for estimate, estimate_t60, estimate_dfn, label in zip(
            estimates.transpose(0, 1),
            estimates_t60.transpose(0, 1),
            estimates_dfn.transpose(0, 1),
            labels.transpose(0, 1),
        ):
            inds = label >= -30
            maes.append((estimate[inds] - label[inds]).abs().mean())
            maes_t60.append((estimate_t60[inds] - label[inds]).abs().mean())
            maes_dfn.append((estimate_dfn[inds] - label[inds]).abs().mean())

        table = Table(title="Test set performance")
        table.add_column("Octave (Hz)")
        [table.add_column(f"{freq}") for freq in freqs]

        row = ["MAE (Proposed) (dB)"] + [f"{float(mae):.1f}" for mae in maes]
        table.add_row(*row)

        row = ["MAE (T60) (dB)"] + [f"{float(mae):.1f}" for mae in maes_t60]
        table.add_row(*row)

        row = ["MAE (DFN) (dB)"] + [f"{float(mae):.1f}" for mae in maes_dfn]
        table.add_row(*row)

        console = Console()
        console.print("\n")
        console.print(table)

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
