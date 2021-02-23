import numpy as np
import torch
from torch.utils.data import TensorDataset

import datasets
from datasets.cmapss import PairedCMAPSS


class PretrainingDataModuleTemplate:
    def test_build_pairs(self):
        for split in ["dev", "val"]:
            with self.subTest(split=split):
                paired_dataset = self.dataset._get_paired_dataset(split)
                pairs = self._get_pairs(paired_dataset)
                self.assertTrue(
                    np.all(pairs[:, 0] < pairs[:, 1])
                )  # query always after anchor
                self.assertTrue(np.all(pairs[:, 2] <= 1))  # domain label is either 1
                self.assertTrue(np.all(pairs[:, 2] >= 0))  # or zero
                run_start_idx = paired_dataset._run_start_idx
                run_idx_of_pair = self._get_run_idx_of_pair(pairs, run_start_idx)
                query_in_same_run = [
                    run_start_idx[run_idx + 1] > query_idx
                    for run_idx, query_idx in zip(run_idx_of_pair, pairs[:, 1])
                ]
                self.assertTrue(all(query_in_same_run))

    def test_domain_labels(self):
        for split in ["dev", "val"]:
            with self.subTest(split=split):
                paired_dataset = self.dataset._get_paired_dataset(split)
                pairs = self._get_pairs(paired_dataset)
                run_start_idx = paired_dataset._run_start_idx
                run_idx_of_pair = self._get_run_idx_of_pair(pairs, run_start_idx)
                run_domain_idx = paired_dataset._run_domain_idx

                expected_domain_labels = [
                    run_domain_idx[run_idx] for run_idx in run_idx_of_pair
                ]
                actual_domain_labels = pairs[:, 2].tolist()
                self.assertListEqual(expected_domain_labels, actual_domain_labels)

    def _get_run_idx_of_pair(self, pairs, run_start_idx):
        """Run idx is number of start idx smaller than or equal to anchor minus one."""
        return np.sum(run_start_idx[:, None] <= pairs[:, 0], axis=0) - 1

    def test_min_distance(self):
        dataset = datasets.PretrainingAdaptionDataModule(
            3, 1, num_samples=10000, min_distance=30, batch_size=16
        )
        dataset.prepare_data()
        dataset.setup()

        for split in ["dev", "val"]:
            with self.subTest(split=split):
                pairs = self._get_pairs(dataset._get_paired_dataset(split))
                distances = pairs[:, 1] - pairs[:, 0]
                self.assertTrue(np.all(pairs[:, 0] < pairs[:, 1]))
                self.assertTrue(np.all(distances >= 30))

    def _get_pairs(self, paired_dataset):
        pairs = np.array(
            [paired_dataset._get_pair_idx() for _ in range(paired_dataset.num_samples)]
        )

        return pairs

    def test_data_structure(self):
        with self.subTest(split="dev"):
            dataloader = self.dataset.train_dataloader()
            self._check_paired_dataset(dataloader.dataset)

        with self.subTest(split="val"):
            loaders = self.dataset.val_dataloader()
            self.assertIsInstance(loaders, list)
            self.assertEqual(self.expected_num_val_loaders, len(loaders))
            self._check_paired_dataset(loaders[0].dataset)
            for dataloader in loaders[1:]:
                self._check_tensor_dataset(dataloader.dataset)

    def _check_paired_dataset(self, data):
        self.assertIsInstance(data, PairedCMAPSS)
        self._check_paired_shapes(data)

    def _check_paired_shapes(self, data):
        for anchors, queries, distances, domain_labels in data:
            self.assertEqual(torch.Size((14, self.window_size)), anchors.shape)
            self.assertEqual(torch.Size((14, self.window_size)), queries.shape)
            self.assertEqual(torch.Size(()), distances.shape)
            self.assertEqual(torch.Size(()), domain_labels.shape)

    def _check_tensor_dataset(self, data):
        self.assertIsInstance(data, TensorDataset)
        self._check_cmapss_shapes(data)

    def _check_cmapss_shapes(self, data):
        for i in range(len(data)):
            features, labels = data[i]
            self.assertEqual(torch.Size((14, self.window_size)), features.shape)
            self.assertEqual(torch.Size(()), labels.shape)

    def test_distances(self):
        with self.subTest(split="dev"):
            _, _, distances, _ = self._run_epoch(self.dataset.train_dataloader())
            self.assertTrue(torch.all(distances >= 0))

        with self.subTest(split="val"):
            _, _, distances, _ = self._run_epoch(self.dataset.val_dataloader()[0])
            self.assertTrue(torch.all(distances >= 0))

    def test_determinism(self):
        with self.subTest(split="dev"):
            train_loader = self.dataset.train_dataloader()
            *one_train_data, one_domain_labels = self._run_epoch(train_loader)
            *another_train_data, another_domain_labels = self._run_epoch(train_loader)

            for one, another in zip(one_train_data, another_train_data):
                self.assertNotEqual(0.0, torch.sum(one - another))

        with self.subTest(split="val"):
            paired_val_loader = self.dataset.val_dataloader()[0]
            one_train_data = self._run_epoch(paired_val_loader)
            another_train_data = self._run_epoch(paired_val_loader)

            for one, another in zip(one_train_data, another_train_data):
                self.assertEqual(0.0, torch.sum(one - another))

    def _run_epoch(self, loader):
        anchors = torch.empty((len(loader.dataset), 14, self.window_size))
        queries = torch.empty((len(loader.dataset), 14, self.window_size))
        distances = torch.empty(len(loader.dataset))
        domain_labels = torch.empty(len(loader.dataset))

        start = 0
        end = loader.batch_size
        for anchor, query, dist, domain in loader:
            anchors[start:end] = anchor
            queries[start:end] = query
            distances[start:end] = dist
            domain_labels[start:end] = domain
            start = end
            end += anchor.shape[0]

        return anchors, queries, distances, domain_labels
