import logging
import os
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import soundfile as sf
import torch
from einops import rearrange
from einops import repeat
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger

from fmtransfer.utils.audio import chunked_follower
from pydx7 import dx7_synth
from pydx7.dx7tools import load_patch

# Setup logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class DynamicRangeTestCallback(Callback):
    """
    Generate plots on the test set results.
    TODO: Modifies the velocity input signal for dynamic range
        verification.
    NOTE: So far, assumes linear envelope generation.
    """

    def __init__(self):
        log.info("Hello from Dynamic Range Test Callback")
        self.env = []  # Target Envelopes
        self.ol = []  # Predicted envelopes
        self.note_contour = []  # Midi Note Contour
        self.vel_contour = []  # Velocity Contour

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        conditioning, ground_truth = batch
        self.env.append(ground_truth.cpu())
        self.note_contour.append(conditioning[:, :, 0].cpu())
        self.vel_contour.append(conditioning[:, :, 1].cpu())
        outlevels, loss = outputs
        self.ol.append(outlevels.cpu())

    def on_test_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if pl_module.envelope_type != "linear":
            log.info("Only supports linear envelope generation")
            return

        envelopes = torch.cat(self.env, dim=0).numpy()
        outlevels = torch.cat(self.ol, dim=0).numpy()
        nc = torch.cat(self.note_contour, dim=0).numpy()
        vel_contour = torch.cat(self.vel_contour, dim=0).numpy()

        # Check for envelope dataset.
        filepath = Path(pl_module.data_dir + pl_module.data_file)
        # Confirm that dataset exists
        if not filepath.exists():
            raise FileNotFoundError(
                f"[DYNAMIC RANGE CALLBACK] Envelope dataset not found for \
                exporting patch. Expected: {filepath}"
            )

        # Define output directory
        if isinstance(trainer.logger, WandbLogger):
            group = trainer.logger._wandb_init.get("group")
            outdir = (
                Path(trainer.logger.save_dir).joinpath("dynamics").joinpath(f"{group}")
            )
            outdir.mkdir(parents=True, exist_ok=True)
        else:
            outdir = Path("./").joinpath("dynamics")
            outdir.mkdir(parents=True, exist_ok=True)

        # Load the patch from module
        dset_dict = torch.load(filepath)
        packed_patch = torch.ByteTensor(dset_dict["patch"])
        specs = load_patch(packed_patch.numpy())

        # Rearrange
        ol = rearrange(outlevels, "b l o -> (b l) o")
        env = rearrange(envelopes, "b l o -> (b l) o")
        nc = rearrange(nc, "b l -> (b l)")
        vel_contour = rearrange(vel_contour, "b l -> (b l)")

        # Synthesize
        synth = dx7_synth(specs, sr=44100, block_size=64)
        f0 = 440 * 2 ** (((nc * 127) - 69) / 12)
        # Denormalize envelopes and render (assumes linear envelopes)
        y = synth.render_from_osc_envelopes(f0, 2 * env)
        y_hat = synth.render_from_osc_envelopes(f0, 2 * ol)

        audio_envelope = self.follow_envelope(y, 44100, 64)
        print(f"[INFO] Velocity Contour is {vel_contour.shape}")
        print(f"[INFO] Audio Envelope is {audio_envelope.shape}")
        print(f"[INFO] Audio Envelope max is {max(audio_envelope)}")
        print(f"[INFO] Audio Envelope min is {min(audio_envelope)}")

        # Normalization to assess dynamic range
        audio_envelope = audio_envelope / max(audio_envelope)

        # Resynth process
        conditioning = torch.stack(
            [torch.FloatTensor(nc), torch.FloatTensor(audio_envelope)], axis=1
        )
        conditioning = repeat(conditioning, "l o -> b l o", b=1)
        conditioning = conditioning.to(pl_module.device)
        resynth_ol = pl_module(conditioning)[0]
        print(f"[INFO] Resynth Osc Envelopes is {resynth_ol.shape}")
        resynth_ol = rearrange(resynth_ol, "1 l o -> l o").cpu().numpy()
        # y_resynthesis = synth.render_from_osc_envelopes(f0, 2 * ol)
        sf.write(f'{outdir}/{specs["name"]}_ref.wav', y, 44100)
        sf.write(f'{outdir}/{specs["name"]}_pred.wav', y_hat, 44100)

        # Plot test set results
        self.plot_results(
            outdir,
            specs["name"],
            vel_contour[0:10000],
            nc[0:10000],
            env[0:10000, :],
            ol[0:10000, :],
        )

        self.plot_results(
            outdir,
            f'{specs["name"]}_resynth',
            audio_envelope[0:10000],
            nc[0:10000],
            env[0:10000, :],
            resynth_ol[0:10000, :],
        )

        return

    def follow_envelope(self, data, rate, block_size):
        indices, envelope = chunked_follower(
            data, rate, attack_factor=100, release_factor=100, chunk_size=block_size
        )
        return np.array(envelope)

    def plot_results(self, outdir, exp_name, vel_contour, nc, env, ol=None):
        """
        Plot envelopes and store in file
        """
        font = {"family": "sans", "weight": "normal", "size": 16}
        matplotlib.rc("font", **font)

        f, axs = plt.subplots(8, 1, sharex=True, figsize=(8, 8))

        axs[0].set_title("$a$")
        axs[0].plot(vel_contour)
        axs[0].grid(True)
        axs[1].set_title("$f$")
        axs[1].plot(nc)
        axs[1].grid(True)

        for i in range(6):
            axs[i + 2].set_title("$ol_{}$".format(i + 1))
            axs[i + 2].plot(env[:, i])
            axs[i + 2].plot(ol[:, i])
            axs[i + 2].grid(True)
        f.tight_layout()

        f.text(0.02, 0.5, "Amplitude", va="center", rotation="vertical")
        f.text(0.5, 0.02, "Frames", ha="center")

        plt.subplots_adjust(
            left=0.14, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.6
        )

        plt.savefig(f"{outdir.absolute()}/{exp_name}.png")
