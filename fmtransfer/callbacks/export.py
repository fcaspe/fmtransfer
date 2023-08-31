import logging
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from torch.utils.mobile_optimizer import optimize_for_mobile

from pydx7.dx7tools import load_patch
from pydx7.dx7tools import unpack_packed_patch

# Setup logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class TS_ExportModelCallback(Callback):
    """
    Callback to export model to TorchScript when finished.

    Traces the model and generates a script.
    Then it wraps this script around a ScriptWrapper that contains training patch,
    and may contain also output amplitude denormalization.
    """

    @torch.no_grad()
    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        log.info(
            f"Export to TorchScript Callback. Envelope type: {pl_module.envelope_type}"
        )

        pl_module = pl_module.to("cpu")
        example_x = torch.ones([1, 1, 2])
        example_state = torch.zeros(1, 1, pl_module.decoder.rnn.hidden_size)
        # Tuple to feed tracer.
        example = (example_x, example_state)

        traced_script_module = optimize_for_mobile(
            torch.jit.trace(pl_module, example, check_trace=True)
        )
        print(traced_script_module.code)

        filepath = Path(pl_module.data_dir + pl_module.data_file)
        # Confirm that dataset exists
        if not filepath.exists():
            raise FileNotFoundError(
                f"Envelope dataset not found for exporting patch. Expected: {filepath}"
            )

        # Load the patch from file and store it in wrapper
        dset_dict = torch.load(filepath)

        packed_patch = torch.ByteTensor(dset_dict["patch"])
        patch_name = load_patch(packed_patch.numpy())["name"]

        unpacked_patch = unpack_packed_patch(dset_dict["patch"])
        if pl_module.envelope_type == "linear":
            script_wrapper = ScriptWrapper_linear(
                traced_script_module,
                torch.ByteTensor(unpacked_patch),
            )
        elif pl_module.envelope_type == "doubling_log":
            script_wrapper = ScriptWrapper_doubling_log(
                traced_script_module,
                torch.ByteTensor(unpacked_patch),
            )

        if isinstance(trainer.logger, WandbLogger):
            group = trainer.logger._wandb_init.get("group")
            outdir = (
                Path(trainer.logger.save_dir).joinpath("export").joinpath(f"{group}")
            )
            outdir.mkdir(parents=True, exist_ok=True)
            # traced_script_module.save(
            #    str(outdir.joinpath(f"{trainer.logger._name}.ts"))
            # )
            script_wrapper.save(str(outdir.joinpath(f"{patch_name}.ts")))
        else:
            outdir = Path("./").joinpath("export")
            outdir.mkdir(parents=True, exist_ok=True)
            script_wrapper.save(f"{patch_name}.ts")
        return


class ScriptWrapper_linear(torch.jit.ScriptModule):
    """
    We can use this wrapper to multiply the output by the correct values
    And to store the training patch
    """

    def __init__(self, script, patch):
        super().__init__()
        self.register_buffer("patch", patch)
        self.script = script

    @torch.jit.script_method
    def forward(self, input: torch.Tensor, state: torch.Tensor):
        outlevels, next_state = self.script(input, state)
        outlevels_denorm = 2 * outlevels
        return (outlevels_denorm, next_state)


class ScriptWrapper_doubling_log(torch.jit.ScriptModule):
    """
    We can use this wrapper to multiply the output by the correct values
    And to store the training patch
    """

    def __init__(self, script, patch):
        super().__init__()
        self.register_buffer("patch", patch)
        self.script = script

    @torch.jit.script_method
    def forward(self, input: torch.Tensor, state: torch.Tensor):
        outlevels, next_state = self.script(input, state)
        outlevels_denorm = 2 ** (outlevels * (1 << 4) - 15)
        return (outlevels_denorm, next_state)
