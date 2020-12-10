from typing import Optional, Union, List

import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

from datasets.cmapss import CMAPSSDataModule, PairedCMAPSS


class DomainAdaptionDataModule(pl.LightningDataModule):
    def __init__(self,
                 fd_source,
                 fd_target,
                 batch_size,
                 max_rul=125,
                 window_size=30,
                 percent_fail_runs=None,
                 percent_broken=None,
                 feature_select=None):
        super().__init__()

        self.fd_source = fd_source
        self.fd_target = fd_target
        self.batch_size = batch_size
        self.window_size = window_size
        self.max_rul = max_rul
        self.percent_broken = percent_broken
        self.percent_fail_runs = percent_fail_runs
        self.feature_select = feature_select

        self.hparams = {'fd_source': self.fd_source,
                        'fd_target': self.fd_target,
                        'batch_size': self.batch_size,
                        'window_size': self.window_size,
                        'max_rul': self.max_rul,
                        'percent_broken': self.percent_broken,
                        'percent_fail_runs': self.percent_fail_runs}

        self.source = CMAPSSDataModule(fd_source, batch_size, max_rul, window_size,
                                       None, None, feature_select)
        self.target = CMAPSSDataModule(fd_target, batch_size, max_rul, window_size,
                                       percent_fail_runs, percent_broken, feature_select)

    def prepare_data(self, *args, **kwargs):
        self.source.prepare_data(*args, **kwargs)
        self.target.prepare_data(*args, **kwargs)

    def setup(self, stage: Optional[str] = None):
        self.source.setup(stage)
        self.target.setup(stage)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self._to_dataset('dev', use_target_labels=False),
                          batch_size=self.batch_size,
                          shuffle=True,
                          pin_memory=True)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return [self.source.val_dataloader(*args, **kwargs), self.target.val_dataloader(*args, **kwargs)]

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return [self.source.test_dataloader(*args, **kwargs), self.target.test_dataloader(*args, **kwargs)]

    def _to_dataset(self, split, use_target_labels):
        source, source_labels = self.source.data[split]
        target, target_labels = self.target.data[split]

        source, source_labels, target, target_labels = _unify_source_and_target_length(source,
                                                                                       source_labels,
                                                                                       target,
                                                                                       target_labels)

        if use_target_labels:
            dataset = TensorDataset(source, source_labels, target, target_labels)
        else:
            dataset = TensorDataset(source, source_labels, target)

        return dataset


class PretrainingAdaptionDataModule(pl.LightningDataModule):
    def __init__(self,
                 fd_source,
                 fd_target,
                 num_samples,
                 batch_size,
                 max_rul=125,
                 window_size=30,
                 min_distance=1,
                 percent_fail_runs=None,
                 percent_broken=None,
                 feature_select=None,
                 truncate_target_val=False):
        super().__init__()

        self.fd_source = fd_source
        self.fd_target = fd_target
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.window_size = window_size
        self.min_distance = min_distance
        self.max_rul = max_rul
        self.percent_broken = percent_broken
        self.percent_fail_runs = percent_fail_runs
        self.feature_select = feature_select
        self.truncate_target_val = truncate_target_val

        self.hparams = {'fd_source': self.fd_source,
                        'fd_target': self.fd_target,
                        'num_samples': self.num_samples,
                        'batch_size': self.batch_size,
                        'window_size': self.window_size,
                        'max_rul': self.max_rul,
                        'min_distance': self.min_distance,
                        'percent_broken': self.percent_broken,
                        'percent_fail_runs': self.percent_fail_runs,
                        'truncate_target_val': self.truncate_target_val}

        self.source = CMAPSSDataModule(fd_source, batch_size, max_rul, window_size,
                                       None, None, feature_select)
        self.target = CMAPSSDataModule(fd_target, batch_size, max_rul, window_size,
                                       percent_fail_runs, percent_broken, feature_select, truncate_target_val)

    def prepare_data(self, *args, **kwargs):
        self.source.prepare_data(*args, **kwargs)
        self.target.prepare_data(*args, **kwargs)

    def setup(self, stage: Optional[str] = None):
        self.source.setup(stage)
        self.target.setup(stage)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self._get_paired_dataset('dev'),
                          batch_size=self.batch_size,
                          pin_memory=True)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        combined_loader = DataLoader(self._get_paired_dataset('val'),
                                     batch_size=self.batch_size,
                                     pin_memory=True)
        source_loader = self.source.val_dataloader()
        target_loader = self.target.val_dataloader()

        return [combined_loader, source_loader, target_loader]

    def _get_paired_dataset(self, split):
        deterministic = split == 'val'
        num_samples = 50000 if split == 'val' else self.num_samples
        paired = PairedCMAPSS([self.source, self.target], split, num_samples, self.min_distance, deterministic)

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