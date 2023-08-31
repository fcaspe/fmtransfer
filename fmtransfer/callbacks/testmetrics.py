import logging
import os
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List

import numpy as np
import pytorch_lightning as pl
import soundfile as sf
import torch
from einops import rearrange
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger

from pydx7 import dx7_synth
from pydx7.dx7tools import load_patch

# Setup logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class TestMetricsCallback(Callback):
    """
    Compute test metrics

    NOTE: So far, assumes linear envelope generation.
    """

    def __init__(self):
        log.info("Hello from TestMetrics Callback")
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

        # Define output directory
        if isinstance(trainer.logger, WandbLogger):
            group = trainer.logger._wandb_init.get("group")
            outdir = (
                Path(trainer.logger.save_dir)
                .joinpath("testmetrics")
                .joinpath(f"{group}")
            )
            outdir.mkdir(parents=True, exist_ok=True)
        else:
            outdir = Path("./").joinpath("testmetrics")
            outdir.mkdir(parents=True, exist_ok=True)

        envelopes = torch.cat(self.env, dim=0).numpy()
        outlevels = torch.cat(self.ol, dim=0).numpy()
        nc = torch.cat(self.note_contour, dim=0).numpy()
        vel_contour = torch.cat(self.vel_contour, dim=0).numpy()

        # Param loss (should be similar to the one informed at the end of test set)
        print(f"envelopes size: {envelopes.shape}")
        print(f"outlevels size: {outlevels.shape}")
        print(f"note contour size: {nc.shape}")
        print(f"vel contour size: {vel_contour.shape}")

        param_loss = np.abs((envelopes - outlevels)).mean()

        # Resynthesis
        filepath = Path(pl_module.data_dir + pl_module.data_file)
        # Confirm that dataset exists
        if not filepath.exists():
            raise FileNotFoundError(
                f"[TEST METRICS  CALLBACK] Envelope dataset not found for \
                exporting patch. Expected: {filepath}"
            )
        # Load the patch from module
        dset_dict = torch.load(filepath)
        packed_patch = torch.ByteTensor(dset_dict["patch"])
        specs = load_patch(packed_patch.numpy())
        synth = dx7_synth(specs, sr=44100, block_size=64)

        # Rearrange and denormalize
        ol = 2 * rearrange(outlevels, "b l o -> (b l) o")
        env = 2 * rearrange(envelopes, "b l o -> (b l) o")

        nc = rearrange(nc, "b l -> (b l)")
        f0 = 440 * 2 ** (((nc * 127) - 69) / 12)

        y = synth.render_from_osc_envelopes(f0, env)
        y_hat = synth.render_from_osc_envelopes(f0, ol)

        # Store synthesized audio
        sf.write(f'{outdir}/{specs["name"]}_ref.wav', y, 44100)
        sf.write(f'{outdir}/{specs["name"]}_pred.wav', y_hat, 44100)

        # Audio metrics
        positions = self._get_positions(vel_contour)
        results = self._get_audio_metrics(y, y_hat, positions)
        results["Envelope_L1"] = param_loss

        for k in results.keys():
            self.log(f"test/{k}", results[k])

    def _get_positions(self, vel_contour: torch.Tensor) -> List:
        """
        Extract positions for each test set element:
        1. Start of non-zero vel_contour
        2. Start of decay ramp
        3. End of decay ramp
        """
        nb = vel_contour.shape[0]
        vel_len = vel_contour.shape[1]
        positions = []
        for i in range(nb):
            base_idx = vel_len * i
            entry = vel_contour[i, :]
            idxs = np.nonzero(entry)[0]
            if len(idxs) == 0:
                continue
            # 1. Find first non zero element
            vel_start = idxs[0]
            # 4. Find last nonzero element (v)
            vel_end = idxs[-1]

            # 5. Find last max val element from trapezoid before decay curve
            maxval = max(entry)
            f = np.where(entry == maxval)[0]
            vel_decay_start = f[-1]

            assert vel_start < vel_decay_start
            assert vel_decay_start < vel_end

            positions.append(
                [vel_start + base_idx, vel_decay_start + base_idx, vel_end + base_idx]
            )

        return positions

    def _get_audio_metrics(self, y, y_hat, positions) -> Dict:
        """
        Computes Audio SNR between reference and resynthesis aggregating:
            1. all note attacks (first 100ms for each note)
            2. All note sustains
            3. All note decays (during decay ramp)
        """

        # Lists that collect synchronized audio excerpts from both reference
        # and resynthesis
        audio_100 = [np.empty(0), np.empty(0)]
        audio_middle = [np.empty(0), np.empty(0)]
        audio_ramp = [np.empty(0), np.empty(0)]

        for i, pos in enumerate(positions):
            # Get pos in samples
            s_start = pos[0] * 64  # Note start
            s100_end = s_start + 4110  # 100ms in 44.1kHz samples

            s_ramp_start = pos[1] * 64  # Decay ramp start
            s_end = pos[2] * 64  # Decay ramp end

            audio_100[0] = np.append(audio_100[0], y[s_start:s100_end])
            audio_100[1] = np.append(audio_100[1], y_hat[s_start:s100_end])

            audio_middle[0] = np.append(audio_middle[0], y[s100_end:s_ramp_start])
            audio_middle[1] = np.append(audio_middle[1], y_hat[s100_end:s_ramp_start])

            audio_ramp[0] = np.append(audio_ramp[0], y[s_ramp_start:s_end])
            audio_ramp[1] = np.append(audio_ramp[1], y_hat[s_ramp_start:s_end])

        # The power of a signal is the sum of the absolute squares
        # of its time-domain samples divided by the signal length
        signal100 = np.mean(audio_100[0] ** 2)
        noise100 = np.mean((audio_100[0] - audio_100[1]) ** 2)
        snr100 = 10 * np.log10(signal100 / noise100)

        signalmiddle = np.mean(audio_middle[0] ** 2)
        noisemiddle = np.mean((audio_middle[0] - audio_middle[1]) ** 2)
        snrmiddle = 10 * np.log10(signalmiddle / noisemiddle)

        signalramp = np.mean(audio_ramp[0] ** 2)
        noiseramp = np.mean((audio_ramp[0] - audio_ramp[1]) ** 2)
        snrramp = 10 * np.log10(signalramp / noiseramp)

        snraudio = 10 * np.log10(np.mean(y**2) / np.mean((y - y_hat) ** 2))

        results = {}
        results["SNR_onset"] = torch.FloatTensor([snr100])
        results["SNR_mid"] = torch.FloatTensor([snrmiddle])
        results["SNR_ramp"] = torch.FloatTensor([snrramp])
        results["SNR_overall"] = torch.FloatTensor([snraudio])

        return results
