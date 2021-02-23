import os
import warnings
from typing import List, Tuple

import numpy as np
import sklearn.preprocessing as scalers
import torch


class AbstractLoader:
    def prepare_data(self):
        raise NotImplementedError

    def load_split(self, split: str) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        raise NotImplementedError


class CMAPSSLoader(AbstractLoader):
    _FMT = (
        "%d %d %.4f %.4f %.1f %.2f %.2f %.2f %.2f %.2f %.2f %.2f "
        "%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.4f %.2f %d %d %.2f %.2f %.4f"
    )
    _TRAIN_PERCENTAGE = 0.8
    WINDOW_SIZES = {1: 30, 2: 20, 3: 30, 4: 15}

    def __init__(
        self,
        fd,
        window_size,
        max_rul,
        percent_broken,
        percent_fail_runs,
        feature_select,
        truncate_val,
    ):
        self.fd = fd
        self.window_size = window_size or self.WINDOW_SIZES[self.fd]
        self.max_rul = max_rul
        self.percent_broken = percent_broken
        self.percent_fail_runs = percent_fail_runs
        self.feature_select = feature_select
        self.truncate_val = truncate_val

        self.DATA_ROOT = os.path.join(
            os.path.dirname(__file__), "../..", "data", "CMAPSS"
        )

    def prepare_data(self):
        # Check if training data was already split
        dev_path = self._file_path("dev")
        if not os.path.exists(dev_path):
            warnings.warn(
                f"Training data for FD{self.fd:03d} not yet split into dev and val. Splitting now."
            )
            self._split_fd_train(self._file_path("train"))

    def _split_fd_train(self, train_path):
        train_data = np.loadtxt(train_path)

        # Split into runs
        _, samples_per_run = np.unique(train_data[:, 0], return_counts=True)
        split_idx = np.cumsum(samples_per_run)[:-1]
        train_data = np.split(train_data, split_idx, axis=0)

        split_idx = int(len(train_data) * self._TRAIN_PERCENTAGE)
        dev_data = np.concatenate(train_data[:split_idx])
        val_data = np.concatenate(train_data[split_idx:])

        data_root, train_file = os.path.split(train_path)
        dev_file = train_file.replace("train_", "dev_")
        dev_file = os.path.join(data_root, dev_file)
        np.savetxt(dev_file, dev_data, fmt=self._FMT)
        val_file = train_file.replace("train_", "val_")
        val_file = os.path.join(data_root, val_file)
        np.savetxt(val_file, val_data, fmt=self._FMT)

    def _file_path(self, split):
        return os.path.join(self.DATA_ROOT, self._file_name(split))

    def _file_name(self, split):
        return f"{split}_FD{self.fd:03d}.txt"

    def load_split(self, split: str) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        file_path = self._file_path(split)

        features = self._load_features(file_path)
        if split == "dev" or (split == "val" and self.truncate_val):
            features = self._truncate_features(features)
        features = self._normalize(features)
        features, time_steps = self._remove_time_steps_from_features(features)

        if split == "dev" or split == "val":
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
                run[:, 1] += (
                    len(run) - num_cycles - 1
                )  # Adjust targets to truncated length
                features[i] = run[:num_cycles]

        return features

    def _normalize(self, features):
        """Normalize features with sklearn transform."""
        # Fit scaler on corresponding training split
        train_file = self._file_path("dev")
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
        file_name = f"RUL_FD{self.fd:03d}.txt"
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
            feature_windows = [
                seq[i : (i + self.window_size)] for i in range(0, num_frames)
            ]
            new_features.extend(feature_windows)
            new_targets.append(target)

        lengths = [len(f) for f in features]
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
                features[i] = seq[-self.window_size :]

        features = np.stack(features, axis=0)
        lengths = [1] * len(features)

        return features, lengths
