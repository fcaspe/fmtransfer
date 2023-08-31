"""
Audio datasets
"""
import logging
import os
import random
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from pydx7 import load_patch_from_bulk
from pydx7 import render_envelopes


# Setup logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class EnvelopeDatasetGenerator:
    """
    Class for generating envelopes and the excitation signals.
            min_sensitivity: Minimum velocity sensitivity to impose to FM carriers.
    """

    def __init__(
        self,
        patch_file: str,
        patch_loc: int,
        note_min: int = 0,
        note_delta: int = 127,
        velocity_min: int = 1,
        velocity_delta: int = 126,
        min_sensitivity: Optional[int] = None,
    ):
        self.patch_file = patch_file
        self.patch_loc = patch_loc
        self.note_min = note_min
        self.note_delta = note_delta
        self.velocity_min = velocity_min
        self.velocity_delta = velocity_delta
        self.min_sensitivity = min_sensitivity

    def generate_dataset(
        self,
        data_dir: str,
        instance_len: int,
        n_instances: int,
        n_events_per_instance: List = [1, 2, 3],
    ) -> Dict:
        # Step1: Open patch
        specs = load_patch_from_bulk(
            data_dir + self.patch_file,
            patch_number=self.patch_loc,
            load_from_sysex=True,
        )
        # Apply minimum sensitivity in specs. TODO: It should affect patch's binary too.
        if self.min_sensitivity is not None:
            if self.min_sensitivity < 0 or self.min_sensitivity > 7:
                log.info(
                    f"Error. Invalid min_sensitivity of {self.min_sensitivity}. \
                        Ignoring setting."
                )
            else:
                log.info(f"Applying a minimum sensitivity of {self.min_sensitivity}.")
                for i in range(6):
                    if specs["outmatrix"][i] == 1:
                        if specs["sensitivity"][i] < self.min_sensitivity:
                            specs["sensitivity"][i] = self.min_sensitivity

        instances = []
        for i in tqdm(range(n_instances)):
            instances.append(
                self.generate_instance(
                    specs, instance_len, n_events=random.choice(n_events_per_instance)
                )
            )
        dset_dict = {}
        dset_dict["patch"] = torch.ByteTensor(specs["binary"])
        dset_dict["instances"] = instances

        return dset_dict

    def generate_instance(self, specs: Dict, instance_len: int, n_events: int) -> List:
        len_list = []
        len_counter = instance_len
        # Assign Lengths for each event
        for i in range(n_events):
            len_list.append(instance_len // n_events)
            len_counter -= instance_len // n_events
        len_list[-1] += len_counter

        # Create instance vectors and concat each event
        env = torch.empty([0, 6])
        qenv = torch.empty([0, 6])
        conditioning = torch.empty([0, 2])
        for event_len in len_list:
            event = self.generate_event(specs, event_len)
            env = torch.cat([env, event[0]], dim=0)
            qenv = torch.cat([qenv, event[1]], dim=0)
            conditioning = torch.cat([conditioning, event[2]], dim=0)

        entry = [env, qenv, conditioning]
        return entry.copy()

    def generate_event(self, specs, event_len) -> List:
        frames_on_min = event_len // 3
        frames_off_min = event_len // 3
        delta_n_on = event_len // 10
        delta_n_off = event_len // 10

        # First, make sure there's space for silence rendering.
        assert frames_on_min + delta_n_on + frames_off_min + delta_n_off < event_len

        # TODO: Issue warning if complete rendering cant happen within frame.

        # Step2: Generate a note-velocity pair.
        note_velocity_pair = np.zeros([1, 2], dtype=int)

        note_velocity_pair[0, 0] = int(
            np.round(np.random.uniform() * self.note_delta + self.note_min)
        )
        note_velocity_pair[0, 1] = int(
            np.round(np.random.uniform() * self.velocity_delta + self.velocity_min)
        )

        # Step 3: Compute lengths for note on, note off and
        # lead-in and lead out silence.

        qenvelopes_ratio = np.ones(6)

        len_n_on = int(np.round(np.random.uniform() * delta_n_on + frames_on_min))
        len_n_off = int(np.round(np.random.uniform() * delta_n_off + frames_off_min))
        len_silence = int((event_len - len_n_on - len_n_off) / 2)
        len_silence = [
            len_silence,
            int(event_len - len_n_on - len_n_off - len_silence),
        ]

        # Step 4: Render note
        # 1. Generate pitch contour
        n = np.zeros(len_silence[0])
        n = np.append(n, np.ones(len_n_on + len_n_off) * note_velocity_pair[0, 0])
        n = np.append(n, np.zeros(len_silence[1]))

        # 2. Render EG Curves
        #    Initial silence
        env = np.zeros([len_silence[0], 6])
        qenv = np.zeros([len_silence[0], 6])

        # Call pydx7 for rendering the EGs
        render = render_envelopes(
            specs, note_velocity_pair[0, 1], len_n_on, len_n_off, qenvelopes_ratio
        )
        env = np.append(env, render[0].transpose(1, 0), 0)
        qenv = np.append(qenv, render[1].transpose(1, 0), 0)

        #    Append silence after note
        env = np.append(env, np.zeros([len_silence[1], 6]), 0)
        qenv = np.append(qenv, np.zeros([len_silence[1], 6]), 0)

        # 3. Generate velocity curves with ramp _-\
        v = np.zeros(len_silence[0])
        von = np.ones(len_n_on) * note_velocity_pair[0, 1]

        # Plain ramp

        # Test the envelope signal (render[0]) for values bigger than 0.01
        # OR operation along the six oscillator dimensions. Sum values.
        # The result minus one is the last index where any value > 0.01

        test = render[0].transpose(1, 0)
        test = test[len_n_on:, :]
        test = test > 0.01
        test = np.sum(test, axis=1)

        last_idx = int(np.sum(test > 0))

        voff = np.linspace(note_velocity_pair[0, 1], 0, last_idx)
        if last_idx < len_n_off:
            voff = np.append(voff, np.zeros(len_n_off - last_idx))

        v = np.append(v, np.append(von, voff))
        v = np.append(v, np.zeros(len_silence[1]))

        # Normalize and cast to FloatTensor
        note_max = self.note_min + self.note_delta
        vel_max = self.velocity_min + self.velocity_delta

        n = torch.FloatTensor(n / note_max)
        v = torch.FloatTensor(v / vel_max)

        env = torch.FloatTensor(env / 2)
        qenv = torch.FloatTensor(qenv / (1 << 28))

        conditioning = torch.stack([n, v], axis=1)

        entry = [env, qenv, conditioning]
        return entry.copy()
