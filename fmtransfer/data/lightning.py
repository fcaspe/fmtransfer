"""Provide LightningDataModule wrappers for datasets.
"""
import logging
import os
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Type

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from fmtransfer.data.envelopes import EnvelopesDataset
from fmtransfer.data.generator import EnvelopeDatasetGenerator

# Setup logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class EnvelopesDataModule(pl.LightningDataModule):
    """
    LightningDataModule for the audio dataset. This class is responsible for
    opening a preprocessed dataset, or generating and storing the dataset
    from a patch file if not available.

    Args:
        batch_size: Batch size, defaults to 32
        num_workers: Number of workers, defaults to 0
        data_dir: Directory to extract the dataset from.
        data_file: Path to dataset file. Provide if dataset is already processed.
        instance_len: Expected instance length
        n_instances: Expected number of instances
        n_events_per_instance: List with valid number of events per instance.
        dataset_class: Dataset class, defaults to AudioDataset
        dataset_kwargs: Additional keyword arguments to pass to the dataset class
            constructor, defaults to None
    """

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        data_dir: str = "dataset/",
        data_file: str = "epiano.pt",
        generator: Optional[EnvelopeDatasetGenerator] = None,
        instance_len: int = 1000,
        n_instances: Optional[int] = 1000,
        n_events_per_instance: List = [1, 2, 3],
        envelope_type: str = "linear",
        dataset_class: Type[EnvelopesDataset] = EnvelopesDataset,
        dataset_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.data_file = data_file
        self.generator = generator
        self.instance_len = instance_len
        self.n_instances = n_instances
        self.n_events_per_instance = n_events_per_instance
        self.envelope_type = envelope_type
        self.dataset_cls = dataset_class
        self.dataset_kwargs = dataset_kwargs or {}

    def prepare_data(self) -> None:
        """
        Download and extract the dataset.

        Args:
            use_preprocessed: Whether to data, defaults to True. If
                False, the dataset will be generated again
        """
        if not Path(self.data_dir + self.data_file).exists():
            # Check if a patch file and location are provided
            if self.generator is not None:
                log.info("Dataset Generator provided. Attempting to create dataset.")
                patch_filepath = Path(self.data_dir + self.generator.patch_file)
                # Confirm that patch exists
                if not patch_filepath.exists():
                    raise FileNotFoundError(
                        f"Patch file not found. Expected: {patch_filepath}"
                    )
                if self.n_instances is None:
                    raise ValueError("Number of instances required.")
                self.create_envelope_dataset()

            # Otherwise, loading fails
            else:
                raise FileNotFoundError(
                    f"Dataset file not found. Expected: \
                        {Path(self.data_dir+self.data_file)}"
                )

        log.info("Dataset already exists.")

    def setup(self, stage: str):
        """
        Assign train/val/test datasets for use in dataloaders.

        Args:
            stage: Current stage (fit, validate, test)
        """
        args = [self.data_dir, self.data_file, self.instance_len, self.envelope_type]
        if stage == "fit":
            self.train_dataset = self.dataset_cls(
                *args,
                split="train",
                **self.dataset_kwargs,
            )
            self.val_dataset = self.dataset_cls(
                *args,
                split="val",
                **self.dataset_kwargs,
            )
        elif stage == "validate":
            self.val_dataset = self.dataset_cls(
                *args,
                split="val",
                **self.dataset_kwargs,
            )
        elif stage == "test":
            self.test_dataset = self.dataset_cls(
                *args,
                split="test",
                **self.dataset_kwargs,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def create_envelope_dataset(self) -> None:
        """
        Create an envelope dataset.
        """
        log.info("Creating dataset...")
        dset_dict = self.generator.generate_dataset(
            self.data_dir,
            self.instance_len,
            self.n_instances,
            self.n_events_per_instance,
        )

        torch.save(dset_dict, Path(self.data_dir + self.data_file))

        return
