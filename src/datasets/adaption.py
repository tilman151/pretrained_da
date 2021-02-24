from typing import List, Optional, Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from datasets.cmapss import AdaptionDataset, CMAPSSDataModule, PairedCMAPSS


class DomainAdaptionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        fd_source,
        fd_target,
        batch_size,
        max_rul=125,
        window_size=None,
        percent_fail_runs=None,
        percent_broken=None,
        feature_select=None,
    ):
        super().__init__()

        self.fd_source = fd_source
        self.fd_target = fd_target
        self.batch_size = batch_size
        self.max_rul = max_rul
        self.percent_broken = percent_broken
        self.percent_fail_runs = percent_fail_runs
        self.feature_select = feature_select

        self.target = CMAPSSDataModule(
            self.fd_target,
            self.batch_size,
            self.max_rul,
            window_size,
            self.percent_fail_runs,
            self.percent_broken,
            self.feature_select,
        )
        self.window_size = self.target.window_size

        self.source = CMAPSSDataModule(
            self.fd_source,
            self.batch_size,
            self.max_rul,
            self.window_size,
            None,
            None,
            self.feature_select,
        )
        self.target_truncated = CMAPSSDataModule(
            self.fd_target,
            self.batch_size,
            self.max_rul,
            self.window_size,
            self.percent_fail_runs,
            self.percent_broken,
            self.feature_select,
            truncate_val=True,
        )

        self.hparams = {
            "fd_source": self.fd_source,
            "fd_target": self.fd_target,
            "batch_size": self.batch_size,
            "window_size": self.window_size,
            "max_rul": self.max_rul,
            "percent_broken": self.percent_broken,
            "percent_fail_runs": self.percent_fail_runs,
        }

    def prepare_data(self, *args, **kwargs):
        self.source.prepare_data(*args, **kwargs)
        self.target.prepare_data(*args, **kwargs)
        self.target_truncated.prepare_data(*args, **kwargs)

    def setup(self, stage: Optional[str] = None):
        self.source.setup(stage)
        self.target.setup(stage)
        self.target_truncated.setup(stage)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self._to_dataset("dev"),
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return [
            self.source.val_dataloader(*args, **kwargs),
            self.target.val_dataloader(*args, **kwargs),
            DataLoader(
                self._get_paired_dataset(), batch_size=self.batch_size, pin_memory=True
            ),
        ]

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return [
            self.source.test_dataloader(*args, **kwargs),
            self.target.test_dataloader(*args, **kwargs),
        ]

    def _to_dataset(self, split):
        source = self.source.to_dataset(split)
        target = self.target.to_dataset(split)
        dataset = AdaptionDataset(source, target)

        return dataset

    def _get_paired_dataset(self):
        deterministic = True
        num_samples = 25000
        min_distance = 1
        paired = PairedCMAPSS(
            [self.target_truncated], "val", num_samples, min_distance, deterministic
        )

        return paired


class PretrainingAdaptionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        fd_source,
        fd_target,
        num_samples,
        batch_size,
        max_rul=125,
        window_size=None,
        min_distance=1,
        percent_fail_runs=None,
        percent_broken=None,
        feature_select=None,
        truncate_target_val=False,
    ):
        super().__init__()

        self.fd_source = fd_source
        self.fd_target = fd_target
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.window_size = window_size or CMAPSSDataModule.WINDOW_SIZES[self.fd_target]
        self.min_distance = min_distance
        self.max_rul = max_rul
        self.percent_broken = percent_broken
        self.percent_fail_runs = percent_fail_runs
        self.feature_select = feature_select
        self.truncate_target_val = truncate_target_val

        self.hparams = {
            "fd_source": self.fd_source,
            "fd_target": self.fd_target,
            "num_samples": self.num_samples,
            "batch_size": self.batch_size,
            "window_size": self.window_size,
            "max_rul": self.max_rul,
            "min_distance": self.min_distance,
            "percent_broken": self.percent_broken,
            "percent_fail_runs": self.percent_fail_runs,
            "truncate_target_val": self.truncate_target_val,
        }

        self.source = CMAPSSDataModule(
            self.fd_source,
            self.batch_size,
            self.max_rul,
            self.window_size,
            None,
            None,
            self.feature_select,
        )
        self.target = CMAPSSDataModule(
            self.fd_target,
            self.batch_size,
            self.max_rul,
            self.window_size,
            self.percent_fail_runs,
            self.percent_broken,
            self.feature_select,
            self.truncate_target_val,
        )

    def prepare_data(self, *args, **kwargs):
        self.source.prepare_data(*args, **kwargs)
        self.target.prepare_data(*args, **kwargs)

    def setup(self, stage: Optional[str] = None):
        self.source.setup(stage)
        self.target.setup(stage)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self._get_paired_dataset("dev"), batch_size=self.batch_size, pin_memory=True
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        combined_loader = DataLoader(
            self._get_paired_dataset("val"), batch_size=self.batch_size, pin_memory=True
        )
        source_loader = self.source.val_dataloader()
        target_loader = self.target.val_dataloader()

        return [combined_loader, source_loader, target_loader]

    def _get_paired_dataset(self, split):
        deterministic = split == "val"
        num_samples = 50000 if split == "val" else self.num_samples
        paired = PairedCMAPSS(
            [self.source, self.target],
            split,
            num_samples,
            self.min_distance,
            deterministic,
        )

        return paired


def _unify_source_and_target_length(source, source_labels, target, target_labels):
    """Make source and target data the same length."""
    num_source = source.shape[0]
    num_target = target.shape[0]
    if num_source > num_target:
        target = target.repeat(num_source // num_target + 1, 1, 1)[:num_source]
        target_labels = target_labels.repeat(num_source // num_target + 1)[:num_source]
    elif num_source < num_target:
        source = source.repeat(num_target // num_source + 1, 1, 1)[:num_target]
        source_labels = source_labels.repeat(num_target // num_source + 1)[:num_target]

    return source, source_labels, target, target_labels
