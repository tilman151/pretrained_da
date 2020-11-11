import os
import warnings
from typing import Union, List, Optional

import numpy as np
import pytorch_lightning as pl
import sklearn.preprocessing as scalers
import torch
from torch.utils.data import DataLoader, TensorDataset, IterableDataset


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
        self.DATA_ROOT = os.path.join(os.path.dirname(__file__), '../..', 'data', 'CMAPSS')

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
        self.lengths = {}

    def _file_path(self, split):
        return os.path.join(self.DATA_ROOT, self._file_name(split))

    def _file_name(self, split):
        return f'{split}_FD{self.fd:03d}.txt'

    def prepare_data(self, *args, **kwargs):
        # Check if training data was already split
        dev_path = self._file_path('dev')
        if not os.path.exists(dev_path):
            warnings.warn(f'Training data for FD{self.fd:03d} not yet split into dev and val. Splitting now.')
            self._split_fd_train(self._file_path('train'))

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
        *self.data['dev'], self.lengths['dev'] = self._setup_split('dev')
        *self.data['val'], self.lengths['val'] = self._setup_split('val')
        *self.data['test'], self.lengths['test'] = self._setup_split('test')

    def _setup_split(self, split):
        file_path = self._file_path(split)

        features = self._load_features(file_path)
        if split == 'dev':
            features = self._truncate_features(features)
        features = self._normalize(features)
        features, time_steps = self._remove_time_steps_from_features(features)

        if split == 'dev' or split == 'val':
            # Build targets from time steps on training
            targets = self._generate_targets(time_steps)
            # Window data to get uniform sequence lengths
            features, targets, lengths = self._window_data(features, targets)
        else:
            # Load targets from file on test
            targets = self._load_targets()
            # Crop data to get uniform sequence lengths
            features, lengths = self._crop_data(features)

        # Switch to channel first
        features = features.transpose((0, 2, 1))
        features = torch.tensor(features, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)

        return features, targets, lengths

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
        train_file = self._file_path('dev')
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

        lengths = [len(f) - self.window_size + 1 for f in features]
        features = np.stack(new_features, axis=0)
        targets = np.concatenate(new_targets)

        return features, targets, lengths

    def _crop_data(self, features):
        """Crop length of features to specified window size."""
        for i, seq in enumerate(features):
            if seq.shape[0] < self.window_size:
                pad = (self.window_size - seq.shape[0], seq.shape[1])
                features[i] = np.concatenate([np.zeros(pad), seq])
            else:
                features[i] = seq[-self.window_size:]

        features = np.stack(features, axis=0)
        lengths = [1] * len(features)

        return features, lengths

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


class BaselineDataModule(pl.LightningDataModule):
    def __init__(self,
                 fd_source,
                 batch_size,
                 max_rul=125,
                 window_size=30,
                 feature_select=None):
        super().__init__()

        self.fd_source = fd_source
        self.batch_size = batch_size
        self.window_size = window_size
        self.max_rul = max_rul
        self.feature_select = feature_select

        self.hparams = {'fd_source': self.fd_source,
                        'batch_size': self.batch_size,
                        'window_size': self.window_size,
                        'max_rul': self.max_rul}

        self.cmapss = {}
        for fd in range(1, 5):
            self.cmapss[fd] = CMAPSSDataModule(fd, batch_size, max_rul, window_size,
                                               None, None, feature_select)

    def prepare_data(self, *args, **kwargs):
        for cmapss_fd in self.cmapss.values():
            cmapss_fd.prepare_data(*args, **kwargs)

    def setup(self, stage: Optional[str] = None):
        for cmapss_fd in self.cmapss.values():
            cmapss_fd.setup(stage)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self._to_dataset(self.fd_source, split='dev'),
                          batch_size=self.batch_size,
                          shuffle=True,
                          pin_memory=True)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self._to_dataset(self.fd_source, split='val'),
                          batch_size=self.batch_size,
                          shuffle=False,
                          pin_memory=True)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        test_dataloaders = []
        for fd_target in range(1, 5):
            target_dl = DataLoader(self._to_dataset(self.fd_source, fd_target, 'test'),
                                   batch_size=self.batch_size,
                                   shuffle=False,
                                   pin_memory=True)
            test_dataloaders.append(target_dl)

        return test_dataloaders

    def _to_dataset(self, fd_source, fd_target=None, split=None):
        if split == 'dev' or split == 'val':
            data = self.cmapss[fd_source].data[split]
        elif split == 'test':
            source, source_labels = self.cmapss[fd_source].data[split]
            target, target_labels = self.cmapss[fd_target].data[split]
            data = _unify_source_and_target_length(source, source_labels, target, target_labels)
        else:
            raise ValueError(f'Invalid split {split}')

        dataset = TensorDataset(*data)

        return dataset


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


class PretrainingDataModule(pl.LightningDataModule):
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
                 feature_select=None):
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

        self.hparams = {'fd_source': self.fd_source,
                        'fd_target': self.fd_target,
                        'num_samples': self.num_samples,
                        'batch_size': self.batch_size,
                        'window_size': self.window_size,
                        'max_rul': self.max_rul,
                        'min_distance': self.min_distance,
                        'percent_broken': self.percent_broken,
                        'percent_fail_runs': self.percent_fail_runs}

        self.source = CMAPSSDataModule(fd_source, batch_size, max_rul, window_size,
                                       None, None, feature_select)
        self.target = CMAPSSDataModule(fd_target, batch_size, max_rul, window_size,
                                       percent_fail_runs, percent_broken, feature_select)

        self.source_pairs = None
        self.target_pairs = None

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
        determinsistic = split == 'val'
        num_samples = 50000 if split == 'val' else self.num_samples
        paired = PairedCMAPSS([self.source, self.target], split, num_samples, self.min_distance, determinsistic)

        return paired


class PairedCMAPSS(IterableDataset):
    def __init__(self, datasets, split, num_samples, min_distance, deterministic=False):
        super().__init__()

        self.datasets = datasets
        self.split = split
        self.min_distance = min_distance
        self.num_samples = num_samples
        self.deterministic = deterministic

        run_lengths = [(length, domain_idx)
                       for domain_idx, dataset in enumerate(self.datasets)
                       for length in dataset.lengths[self.split]]
        self._run_start_idx = np.cumsum([length for length, _ in run_lengths if length > self.min_distance])
        self._run_idx = np.arange(len(self._run_start_idx) - 1)
        self._run_domain_idx = [domain_idx for _, domain_idx in run_lengths]

        self._features = torch.cat([dataset.data[self.split][0] for dataset in self.datasets])
        self._max_rul = max(dataset.max_rul for dataset in self.datasets)

        self._current_iteration = 0
        self._rng = self._reset_rng()

    def _reset_rng(self):
        return np.random.default_rng(seed=42)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        self._current_iteration = 0
        if self.deterministic:
            self._rng = self._reset_rng()

        return self

    def __next__(self):
        if self._current_iteration < self.num_samples:
            self._current_iteration += 1
            pair_idx = self._get_pair_idx()
            return self._build_pair(pair_idx)
        else:
            raise StopIteration

    def _get_pair_idx(self):
        chosen_run_idx = self._rng.choice(self._run_idx)
        domain_label = self._run_domain_idx[chosen_run_idx]
        anchor_idx = self._rng.integers(low=self._run_start_idx[chosen_run_idx],
                                        high=self._run_start_idx[chosen_run_idx + 1] - self.min_distance)
        query_idx = self._rng.integers(low=anchor_idx + self.min_distance,
                                       high=self._run_start_idx[chosen_run_idx + 1])

        return anchor_idx, query_idx, domain_label

    def _build_pair(self, pair_idx):
        anchors = self._features[pair_idx[0]]
        queries = self._features[pair_idx[1]]
        domain_label = torch.tensor(pair_idx[2], dtype=torch.float)
        distances = torch.tensor(pair_idx[1] - pair_idx[0], dtype=torch.float) / self._max_rul
        distances = torch.clamp_max(distances, max=1)  # max distance is max_rul

        return anchors, queries, distances, domain_label


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
