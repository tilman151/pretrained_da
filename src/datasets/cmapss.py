import os
from typing import List, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, IterableDataset, TensorDataset

from datasets.loader import CMAPSSLoader


class CMAPSSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        fd,
        batch_size,
        max_rul=125,
        window_size=None,
        percent_fail_runs=None,
        percent_broken=None,
        feature_select=None,
        truncate_val=False,
    ):
        super().__init__()

        self.fd = fd
        self.batch_size = batch_size
        self.window_size = window_size
        self.max_rul = max_rul
        self.percent_broken = percent_broken
        self.percent_fail_runs = percent_fail_runs
        self.feature_select = feature_select
        self.truncate_val = truncate_val

        self.hparams = {
            "fd": self.fd,
            "batch_size": self.batch_size,
            "window_size": self.window_size,
            "max_rul": self.max_rul,
            "percent_broken": self.percent_broken,
            "percent_fail_runs": self.percent_fail_runs,
            "truncate_val": self.truncate_val,
        }

        self.data = {}
        self.lengths = {}
        self._loader = CMAPSSLoader(
            self.fd,
            self.window_size,
            self.max_rul,
            self.percent_broken,
            self.percent_fail_runs,
            self.feature_select,
            self.truncate_val,
        )

    def prepare_data(self, *args, **kwargs):
        self._loader.prepare_data()

    def setup(self, stage: Optional[str] = None):
        *self.data["dev"], self.lengths["dev"] = self._setup_split("dev")
        *self.data["val"], self.lengths["val"] = self._setup_split("val")
        *self.data["test"], self.lengths["test"] = self._setup_split("test")

    def _setup_split(self, split):
        features, targets = self._loader.load_split(split)
        lengths = [len(f) for f in features]
        features = torch.cat(features)
        targets = torch.cat(targets)

        return features, targets, lengths

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self._to_dataset(*self.data["dev"]),
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self._to_dataset(*self.data["val"]),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self._to_dataset(*self.data["test"]),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )

    def _to_dataset(self, features, targets):
        return TensorDataset(features, targets)


class PairedCMAPSS(IterableDataset):
    def __init__(
        self,
        datasets,
        split,
        num_samples,
        min_distance,
        deterministic=False,
        labeled=False,
    ):
        super().__init__()

        self.datasets = datasets
        self.split = split
        self.min_distance = min_distance
        self.num_samples = num_samples
        self.deterministic = deterministic

        if not all(d.window_size == self.datasets[0].window_size for d in self.datasets):
            window_sizes = [d.window_size for d in self.datasets]
            raise ValueError(
                f"Datasets to be paired do not have the same window size, but {window_sizes}"
            )

        self._run_start_idx = None
        self._run_idx = None
        self._run_domain_idx = None
        self._features = None
        self._labels = None
        self._prepare_datasets()

        self._max_rul = max(dataset.max_rul for dataset in self.datasets)
        self._current_iteration = 0
        self._rng = self._reset_rng()
        self._get_pair_func = (
            self._get_labeled_pair_idx if labeled else self._get_pair_idx
        )

    def _prepare_datasets(self):
        run_start_idx = [0]
        run_idx = []
        run_domain_idx = []
        features = []
        labels = []
        for domain_idx, dataset in enumerate(self.datasets):
            run_features, run_labels = dataset.data[self.split]
            run_lengths = dataset.lengths[self.split]
            run_features = torch.split(run_features, run_lengths)
            run_labels = torch.split(run_labels, run_lengths)
            for i, length in enumerate(run_lengths):
                if length > self.min_distance:
                    run_start_idx.append(run_start_idx[-1] + length)
                    run_idx.append(len(run_idx))
                    run_domain_idx.append(domain_idx)
                    features.append(run_features[i])
                    labels.append(run_labels[i])

        self._run_start_idx = np.array(run_start_idx)
        self._run_idx = np.array(run_idx)
        self._run_domain_idx = np.array(run_domain_idx)
        self._features = torch.cat(features)
        self._labels = torch.cat(labels).numpy()

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
            pair_idx = self._get_pair_func()
            return self._build_pair(pair_idx)
        else:
            raise StopIteration

    def _get_pair_idx(self):
        chosen_run_idx = self._rng.choice(self._run_idx)
        domain_label = self._run_domain_idx[chosen_run_idx]
        middle_idx = (
            self._run_start_idx[chosen_run_idx + 1] + self._run_start_idx[chosen_run_idx]
        ) // 2
        anchor_idx = self._rng.integers(
            low=self._run_start_idx[chosen_run_idx],
            high=self._run_start_idx[chosen_run_idx + 1] - self.min_distance,
        )
        end_idx = (
            middle_idx
            if anchor_idx < (middle_idx - self.min_distance)
            else self._run_start_idx[chosen_run_idx + 1]
        )
        query_idx = self._rng.integers(
            low=anchor_idx + self.min_distance,
            high=end_idx,
        )
        distance = query_idx - anchor_idx if anchor_idx > middle_idx else 0

        return anchor_idx, query_idx, domain_label, distance

    def _get_labeled_pair_idx(self):
        chosen_run_idx = self._rng.choice(self._run_idx)
        domain_label = self._run_domain_idx[chosen_run_idx]
        anchor_idx = self._rng.integers(
            low=self._run_start_idx[chosen_run_idx],
            high=self._run_start_idx[chosen_run_idx + 1] - self.min_distance,
        )
        query_idx = self._rng.integers(
            low=anchor_idx + self.min_distance,
            high=self._run_start_idx[chosen_run_idx + 1],
        )
        # RUL label difference is negative time step difference
        distance = self._labels[anchor_idx] - self._labels[query_idx]

        return anchor_idx, query_idx, domain_label, distance

    def _build_pair(self, pair_idx):
        anchors = self._features[pair_idx[0]]
        queries = self._features[pair_idx[1]]
        domain_label = torch.tensor(pair_idx[2], dtype=torch.float)
        distances = torch.tensor(pair_idx[3], dtype=torch.float) / self._max_rul
        distances = torch.clamp_max(distances, max=1)  # max distance is max_rul

        return anchors, queries, distances, domain_label
