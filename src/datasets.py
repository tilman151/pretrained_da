import os
import warnings
from typing import Any, Union, List, Optional

import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
import sklearn.preprocessing as scalers


class CMAPSSDataModule(pl.LightningDataModule):
    def __init__(self,
                 fd,
                 batch_size,
                 max_rul=125,
                 window_size=30,
                 percent_fail_runs=None,
                 percent_broken=None,
                 feature_select=None):
        super().__init__()
        self.DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', 'data', 'CMAPSS')

        # Select features according to https://doi.org/10.1016/j.ress.2017.11.021
        if feature_select is None:
            feature_select = [4, 5, 6, 9, 10, 11, 13, 14, 15, 16, 17, 19, 22, 23]

        self.fd = fd
        self.batch_size = batch_size
        self.window_size = window_size
        self.max_rul = max_rul
        self.percent_broken = percent_broken
        self.percent_fail_runs = percent_fail_runs
        self.feature_select = feature_select

        self.hparams = {'fd': self.fd,
                        'batch_size': self.batch_size,
                        'window_size': self.window_size,
                        'max_rul': self.max_rul,
                        'percent_broken': self.percent_broken,
                        'percent_fail_runs': self.percent_fail_runs}

        self.data = {}

    def _file_name(self, split):
        return f'{split}_FD{self.fd:03d}.txt'

    def prepare_data(self, *args, **kwargs):
        # Check if training data was already split
        dev_path = os.path.join(self.DATA_ROOT, self._file_name('dev'))
        if not os.path.exists(dev_path):
            warnings.warn(f'Training data for FD{self.fd:03d} not yet split into dev and val. Splitting now.')
            self._split_fd_train(os.path.join(self.DATA_ROOT, self._file_name('train')))

    def _split_fd_train(self, train_path):
        train_percentage = 0.8
        fmt = '%d %d %.4f %.4f %.1f %.2f %.2f %.2f %.2f %.2f %.2f %.2f ' \
              '%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.4f %.2f %d %d %.2f %.2f %.4f'

        train_data = np.loadtxt(train_path)

        # Split into runs
        _, samples_per_run = np.unique(train_data[:, 0], return_counts=True)
        split_idx = np.cumsum(samples_per_run)[:-1]
        train_data = np.split(train_data, split_idx, axis=0)

        split_idx = int(len(train_data) * train_percentage)
        dev_data = np.concatenate(train_data[:split_idx])
        val_data = np.concatenate(train_data[split_idx:])

        data_root, train_file = os.path.split(train_path)
        dev_file = train_file.replace('train_', 'dev_')
        dev_file = os.path.join(data_root, dev_file)
        np.savetxt(dev_file, dev_data, fmt=fmt)
        val_file = train_file.replace('train_', 'val_')
        val_file = os.path.join(data_root, val_file)
        np.savetxt(val_file, val_data, fmt=fmt)

    def setup(self, stage: Optional[str] = None):
        self.data['dev'] = self._setup_split('dev')
        self.data['val'] = self._setup_split('val')
        self.data['test'] = self._setup_split('test')

    def _setup_split(self, split):
        file_path = os.path.join(self.DATA_ROOT, self._file_name(split))

        features = self._load_features(file_path)
        if split == 'dev':
            features = self._truncate_features(features)
        features = self._normalize(features)
        features, time_steps = self._remove_time_steps_from_features(features)

        if split == 'dev' or split == 'val':
            # Build targets from time steps on training
            targets = self._generate_targets(time_steps)
            # Window data to get uniform sequence lengths
            features, targets = self._window_data(features, targets)
        else:
            # Load targets from file on test
            targets = self._load_targets()
            # Crop data to get uniform sequence lengths
            features = self._crop_data(features)

        # Switch to channel first
        features = features.transpose((0, 2, 1))
        features = torch.tensor(features, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)

        return features, targets

    def _load_features(self, file_path):
        features = np.loadtxt(file_path)

        feature_idx = [0, 1] + [idx + 2 for idx in self.feature_select]
        features = features[:, feature_idx]

        # Split into runs
        _, samples_per_run = np.unique(features[:, 0], return_counts=True)
        split_idx = np.cumsum(samples_per_run)[:-1]
        features = np.split(features, split_idx, axis=0)

        return features

    def _truncate_features(self, features):
        # Truncate the number of runs to failure
        if self.percent_fail_runs is not None and self.percent_fail_runs < 1:
            num_runs = int(self.percent_fail_runs * len(features))
            features = features[:num_runs]

        # Truncate the number of samples per run, starting at failure
        if self.percent_broken is not None and self.percent_broken < 1:
            for i, run in enumerate(features):
                num_cycles = int(self.percent_broken * len(run))
                run[:, 1] += len(run) - num_cycles - 1  # Adjust targets to truncated length
                features[i] = run[:num_cycles]

        return features

    def _normalize(self, features):
        """Normalize features with sklearn transform."""
        # Fit scaler on corresponding training split
        train_file = os.path.join(self.DATA_ROOT, self._file_name('dev'))
        train_features = self._load_features(train_file)
        full_features = np.concatenate(train_features, axis=0)
        scaler = scalers.MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(full_features[:, 2:])

        # Normalize features
        for i, run in enumerate(features):
            features[i][:, 2:] = scaler.transform(run[:, 2:])

        return features

    @staticmethod
    def _remove_time_steps_from_features(features):
        """Extract and return time steps from feature array."""
        time_steps = []
        for i, seq in enumerate(features):
            time_steps.append(seq[:, 1])
            seq = seq[:, 2:]
            features[i] = seq

        return features, time_steps

    def _generate_targets(self, time_steps):
        """Generate RUL targets from time steps."""
        return [np.minimum(self.max_rul, steps)[::-1] for steps in time_steps]

    def _load_targets(self):
        """Load target file."""
        file_name = f'RUL_FD{self.fd:03d}.txt'
        file_path = os.path.join(self.DATA_ROOT, file_name)
        targets = np.loadtxt(file_path)

        targets = np.minimum(self.max_rul, targets)

        return targets

    def _window_data(self, features, targets):
        """Window features with specified window size."""
        new_features = []
        new_targets = []
        for seq, target in zip(features, targets):
            num_frames = seq.shape[0]
            seq = np.concatenate([np.zeros((self.window_size - 1, seq.shape[1])), seq])
            feature_windows = [seq[i:(i+self.window_size)] for i in range(0, num_frames)]
            new_features.extend(feature_windows)
            new_targets.append(target)

        features = np.stack(new_features, axis=0)
        targets = np.concatenate(new_targets)

        return features, targets

    def _crop_data(self, features):
        """Crop length of features to specified window size."""
        for i, seq in enumerate(features):
            if seq.shape[0] < self.window_size:
                pad = (self.window_size - seq.shape[0], seq.shape[1])
                features[i] = np.concatenate([np.zeros(pad), seq])
            else:
                features[i] = seq[-self.window_size:]

        features = np.stack(features, axis=0)

        return features

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self._to_dataset(*self.data['dev']),
                          batch_size=self.batch_size,
                          shuffle=True,
                          pin_memory=True)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self._to_dataset(*self.data['val']),
                          batch_size=self.batch_size,
                          shuffle=False,
                          pin_memory=True)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self._to_dataset(*self.data['test']),
                          batch_size=self.batch_size,
                          shuffle=False,
                          pin_memory=True)

    def _to_dataset(self, features, targets):
        return TensorDataset(features, targets)


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
        return DataLoader(self._to_dataset('val', use_target_labels=True),
                          batch_size=self.batch_size,
                          shuffle=False,
                          pin_memory=True)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self._to_dataset('test', use_target_labels=True),
                          batch_size=self.batch_size,
                          shuffle=False,
                          pin_memory=True)

    def _to_dataset(self, split, use_target_labels):
        source, source_labels = self.source.data[split]
        target, target_labels = self.target.data[split]

        # Make source and target data the same length
        num_source = source.shape[0]
        num_target = target.shape[0]
        if num_source > num_target:
            target = target.repeat(num_source // num_target + 1, 1, 1)[:num_source]
            target_labels = target_labels.repeat(num_source // num_target + 1)[:num_source]
        elif num_source < num_target:
            source = source.repeat(num_target // num_source + 1, 1, 1)[:num_target]
            source_labels = source_labels.repeat(num_target // num_source + 1)[:num_target]

        if use_target_labels:
            dataset = TensorDataset(source, source_labels, target, target_labels)
        else:
            dataset = TensorDataset(source, source_labels, target)

        return dataset
