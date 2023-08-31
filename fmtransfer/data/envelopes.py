"""
Audio datasets
"""
import logging
import os
from pathlib import Path
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split


# Setup logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class EnvelopesDataset(Dataset):
    """
    Dataset of envelopes.

    Args:
        data_dir: Path to the directory containing the dataset.
        data_file: Name of the H5 file.
        instance_len: Expected instance len of the envelopes.
        seed: Seed for random number generator used to split the dataset.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        data_file: str,
        instance_len: int,
        envelope_type: str,
        split: Optional[str] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.filepath = Path(data_dir + data_file)
        self.instance_len = instance_len
        self.seed = seed

        if envelope_type not in ["linear", "doubling_log"]:
            raise ValueError(
                "Invalid envelope type. Must be one of 'linear' or 'doubling_log'."
            )
        # log.info(f'Envelope type: {envelope_type}')
        if envelope_type == "linear":
            self.envpos = 0
        elif envelope_type == "doubling_log":
            self.envpos = 1

        # Confirm that preprocessed dataset exists
        if not self.filepath.exists():
            raise FileNotFoundError(
                f"Envelope dataset not found. Expected: {self.data_dir}"
            )

        # Load the file
        dset_dict = torch.load(self.filepath)
        self.instances = dset_dict["instances"]
        self.patch = dset_dict["patch"]
        self._random_split(split)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx) -> Tuple[torch.Tensor]:
        envelope = self.instances[idx][self.envpos]
        conditioning = self.instances[idx][2]

        # Confirm instance len
        assert envelope.shape == (
            self.instance_len,
            6,
        ), f"Incorrect envelope shape. Was {envelope.shape}"
        assert conditioning.shape == (
            self.instance_len,
            2,
        ), f"Incorrect conditioning shape. Was {conditioning.shape}"

        return (conditioning, envelope)

    def _random_split(self, split: str):
        """
        Split the dataset into train, validation, and test sets.

        Args:
            split: Split to return. Must be one of 'train', 'val', or 'test'.
        """

        if split not in ["train", "val", "test"]:
            raise ValueError("Invalid split. Must be one of 'train', 'val', or 'test'.")

        splits = random_split(
            self.instances,
            [0.8, 0.1, 0.1],
            generator=torch.Generator().manual_seed(self.seed),
        )
        # Set the file list based on the split
        if split == "train":
            self.instances = splits[0]
        elif split == "val":
            self.instances = splits[1]
        elif split == "test":
            self.instances = splits[2]
