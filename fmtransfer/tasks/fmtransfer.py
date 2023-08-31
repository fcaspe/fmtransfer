"""
LightningModule for envelope learning
"""
from typing import Callable
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.cli import LRSchedulerCallable
from pytorch_lightning.cli import OptimizerCallable


class FMTransfer(pl.LightningModule):
    """
    LightningModule for control prediction

    Args:
        decoder (nn.Module): Autoregressive / TCN decoder to infer controls
            from an embedding input space.
        loss_fn (Union[Callable, nn.Module]): Loss function to use for training
        optimizer A torch.optimizer
        lr_scheduler A torch.optimizer.lr_scheduler
        lr_scheduler_freq_steps (int): Frequency in steps to update scheduler
        projection (Optional[nn.Module]): Projects decoder embedding
            on a solution space.
        encoder (Optional[nn.Module]): Receives conditioning params and
            generates main embedding
        float32_matmul_precision(Literal["medium", "high", "highest", None]): Sets
            the precision of float32 matmul operations.
        data_dir (Optional[str]): Path to dataset filename dir.
        data_file (Optional[str]): Dataset filename to extract patch from.
        envelope_type (Literal["linear","doubling_log"]): Training Envelope type
    """

    def __init__(
        self,
        decoder: nn.Module,
        loss_fn: Union[Callable, nn.Module],
        optimizer: OptimizerCallable,
        lr_scheduler: LRSchedulerCallable = None,
        lr_scheduler_steps: int = 10000,
        projection: Optional[nn.Module] = None,
        encoder: Optional[nn.Module] = None,
        relay_conditioning_to_projection: bool = False,
        output_activation: Optional[nn.Module] = None,
        float32_matmul_precision: Literal["medium", "high", "highest", None] = None,
        data_dir: Optional[str] = None,
        data_file: Optional[str] = None,
        envelope_type: Literal["linear", "doubling_log"] = "linear",
    ):
        super().__init__()

        self.decoder = decoder
        self.projection = projection
        self.encoder = encoder
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_steps = lr_scheduler_steps
        self.relay_conditioning_to_projection = relay_conditioning_to_projection
        self.output_activation = output_activation
        self.data_dir = data_dir
        self.data_file = data_file
        self.envelope_type = envelope_type
        if float32_matmul_precision is not None:
            torch.set_float32_matmul_precision(float32_matmul_precision)

    def forward(
        self,
        conditioning: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ):
        # Main embeddings
        in_embedding = conditioning
        if self.encoder is not None:
            in_embedding = self.encoder(conditioning)

        if state is not None:
            out_embedding, next_state = self.decoder(in_embedding, state)
        else:
            out_embedding, next_state = self.decoder(in_embedding)

        if self.projection is not None:
            if self.relay_conditioning_to_projection is True:
                out_embedding = torch.cat([out_embedding, conditioning], dim=-1)

            outlevels = self.projection(out_embedding)
        else:
            outlevels = out_embedding

        if self.output_activation is not None:
            outlevels = self.output_activation(outlevels)

        return (outlevels, next_state)

    def _do_step(self, batch: Tuple[torch.Tensor, ...]):
        if len(batch) == 2:
            conditioning, ground_truth = batch
        else:
            raise ValueError("Expected batch to be a tuple of length 2")

        outlevels, state = self(conditioning)
        loss = self.loss_fn(outlevels, ground_truth)
        return outlevels, loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        outlevels, loss = self._do_step(batch)
        self.log("train/loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        outlevels, loss = self._do_step(batch)
        self.log("validation/loss", loss)
        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        outlevels, loss = self._do_step(batch)
        self.log("test/loss", loss)
        return outlevels, loss

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())

        # Create a lr scheduler
        if self.lr_scheduler is not None:
            scheduler = self.lr_scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "step",
                    "frequency": 1,
                    "strict": True,
                },
            }
        return {"optimizer": optimizer}

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        """
        There is a bug while using scheduler.interval = 'step'.
        If we set scheduler frequency inside its configuration, it doesn't work,
        because the step counter goes back to zero for each epoch.
        We override lr_scheduler_step() to update the scheduler

        TODO: If updating, optimizer_idx argument removed in lightning 2.0
        """
        if self.global_step % self.lr_scheduler_steps == 0:
            return super().lr_scheduler_step(scheduler, optimizer_idx, metric)
