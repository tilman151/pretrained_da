from typing import List, Optional, Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

from datasets.adaption import _unify_source_and_target_length
from datasets.cmapss import CMAPSSDataModule, PairedCMAPSS


class BaselineDataModule(pl.LightningDataModule):
    def __init__(
        self,
        fd_source,
        batch_size,
        max_rul=125,
        window_size=None,
        percent_fail_runs=None,
        feature_select=None,
    ):
        super().__init__()

        self.fd_source = fd_source
        self.batch_size = batch_size
        self.max_rul = max_rul
        self.window_size = window_size or CMAPSSDataModule.WINDOW_SIZES[self.fd_source]
        self.percent_fail_runs = percent_fail_runs
        self.feature_select = feature_select

        self.hparams = {
            "fd_source": self.fd_source,
            "batch_size": self.batch_size,
            "window_size": self.window_size,
            "percent_fail_runs": self.percent_fail_runs,
            "max_rul": self.max_rul,
        }

        self.cmapss = {}
        for fd in range(1, 5):
            self.cmapss[fd] = CMAPSSDataModule(
                fd,
                self.batch_size,
                self.max_rul,
                self.window_size,
                self.percent_fail_runs,
                None,
                self.feature_select,
            )

    def prepare_data(self, *args, **kwargs):
        for cmapss_fd in self.cmapss.values():
            cmapss_fd.prepare_data(*args, **kwargs)

    def setup(self, stage: Optional[str] = None):
        for cmapss_fd in self.cmapss.values():
            cmapss_fd.setup(stage)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self._to_dataset(self.fd_source, split="dev"),
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self._to_dataset(self.fd_source, split="val"),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        test_dataloaders = []
        for fd_target in range(1, 5):
            target_dl = DataLoader(
                self._to_dataset(self.fd_source, fd_target, "test"),
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True,
            )
            test_dataloaders.append(target_dl)

        return test_dataloaders

    def _to_dataset(self, fd_source, fd_target=None, split=None):
        if split == "dev" or split == "val":
            data = self.cmapss[fd_source].data[split]
        elif split == "test":
            source, source_labels = self.cmapss[fd_source].data[split]
            target, target_labels = self.cmapss[fd_target].data[split]
            data = _unify_source_and_target_length(
                source, source_labels, target, target_labels
            )
        else:
            raise ValueError(f"Invalid split {split}")

        dataset = TensorDataset(*data)

        return dataset


class PretrainingBaselineDataModule(pl.LightningDataModule):
    def __init__(
        self,
        fd_source,
        num_samples,
        batch_size,
        max_rul=125,
        window_size=None,
        min_distance=1,
        percent_fail_runs=None,
        percent_broken=None,
        feature_select=None,
        truncate_val=False,
    ):
        super().__init__()

        self.fd_source = fd_source
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.window_size = window_size or CMAPSSDataModule.WINDOW_SIZES[self.fd_source]
        self.min_distance = min_distance
        self.max_rul = max_rul
        self.percent_broken = percent_broken
        self.percent_fail_runs = percent_fail_runs
        self.feature_select = feature_select
        self.truncate_val = truncate_val

        self.hparams = {
            "fd_source": self.fd_source,
            "num_samples": self.num_samples,
            "batch_size": self.batch_size,
            "window_size": self.window_size,
            "max_rul": self.max_rul,
            "min_distance": self.min_distance,
            "percent_broken": self.percent_broken,
            "percent_fail_runs": self.percent_fail_runs,
            "truncate_val": self.truncate_val,
        }

        self.source = CMAPSSDataModule(
            fd_source,
            batch_size,
            max_rul,
            window_size,
            percent_fail_runs,
            percent_broken,
            feature_select,
            truncate_val,
        )

    def prepare_data(self, *args, **kwargs):
        self.source.prepare_data(*args, **kwargs)

    def setup(self, stage: Optional[str] = None):
        self.source.setup(stage)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self._get_paired_dataset("dev"), batch_size=self.batch_size, pin_memory=True
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        combined_loader = DataLoader(
            self._get_paired_dataset("val"), batch_size=self.batch_size, pin_memory=True
        )
        source_loader = self.source.val_dataloader()

        return [combined_loader, source_loader]

    def _get_paired_dataset(self, split):
        deterministic = split == "val"
        num_samples = 25000 if split == "val" else self.num_samples
        paired = PairedCMAPSS(
            [self.source], split, num_samples, self.min_distance, deterministic
        )

        return paired
