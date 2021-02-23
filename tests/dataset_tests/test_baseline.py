import unittest

import torch
import torch.utils.data
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler

import datasets
from datasets import cmapss
from datasets.adaption import _unify_source_and_target_length
from tests.dataset_tests.templates import PretrainingDataModuleTemplate


class TestCMAPSSBaseline(unittest.TestCase):
    def setUp(self):
        self.dataset = datasets.BaselineDataModule(
            3, batch_size=16, percent_fail_runs=0.8
        )
        self.dataset.prepare_data()
        self.dataset.setup()

    def test_override_window_size(self):
        dataset = datasets.BaselineDataModule(
            3, batch_size=16, percent_fail_runs=0.8, window_size=40
        )
        dataset.prepare_data()
        dataset.setup()
        train_loader = dataset.train_dataloader()

        batch, _ = next(iter(train_loader))
        self.assertEqual(40, batch.shape[2])

    def test_train_batch_structure(self):
        train_loader = self.dataset.train_dataloader()
        self.assertIsInstance(train_loader.sampler, RandomSampler)
        self._assert_train_val_batch_structure(train_loader)

    def test_val_batch_structure(self):
        val_loader = self.dataset.val_dataloader()
        self.assertIsInstance(val_loader.sampler, SequentialSampler)
        self._assert_train_val_batch_structure(val_loader)

    def test_test_batch_structure(self):
        window_size = datasets.cmapss.CMAPSSDataModule.WINDOW_SIZES[
            self.dataset.fd_source
        ]
        test_loaders = self.dataset.test_dataloader()
        for test_loader in test_loaders:
            self.assertIsInstance(test_loader.sampler, SequentialSampler)
            batch = next(iter(test_loader))
            self.assertEqual(4, len(batch))
            source, source_labels, target, target_labels = batch
            self.assertEqual(torch.Size((16, 14, window_size)), source.shape)
            self.assertEqual(torch.Size((16, 14, window_size)), target.shape)
            self.assertEqual(torch.Size((16,)), source_labels.shape)
            self.assertEqual(torch.Size((16,)), target_labels.shape)

    def _assert_train_val_batch_structure(self, loader):
        batch = next(iter(loader))
        self.assertEqual(2, len(batch))
        features, labels = batch
        self.assertEqual(torch.Size((16, 14, 30)), features.shape)
        self.assertEqual(torch.Size((16,)), labels.shape)

    def test_selected_source_on_train(self):
        fd_source = self.dataset.fd_source
        baseline_train_dataset = self.dataset._to_dataset(fd_source, split="dev")
        source_train_dataset = self.dataset.cmapss[fd_source]._to_dataset(
            *self.dataset.cmapss[fd_source].data["dev"]
        )
        self._assert_datasets_equal(baseline_train_dataset, source_train_dataset)

    def test_selected_source_on_val(self):
        fd_source = self.dataset.fd_source
        baseline_train_dataset = self.dataset._to_dataset(fd_source, split="val")
        source_train_dataset = self.dataset.cmapss[fd_source]._to_dataset(
            *self.dataset.cmapss[fd_source].data["val"]
        )
        self._assert_datasets_equal(baseline_train_dataset, source_train_dataset)

    def test_selected_both_on_test(self):
        fd_source = self.dataset.fd_source
        fd_target = 1
        baseline_train_dataset = self.dataset._to_dataset(
            fd_source, fd_target, split="test"
        )
        combined_data = _unify_source_and_target_length(
            *self.dataset.cmapss[fd_source].data["test"],
            *self.dataset.cmapss[fd_target].data["test"],
        )
        source_train_dataset = TensorDataset(*combined_data)
        self._assert_datasets_equal(baseline_train_dataset, source_train_dataset)

    def test_received_all_datasets_on_test(self):
        test_loaders = self.dataset.test_dataloader()
        fd_source = self.dataset.fd_source
        for fd_target, test_loader in enumerate(test_loaders, start=1):
            test_data = test_loader.dataset
            combined_data = _unify_source_and_target_length(
                *self.dataset.cmapss[fd_source].data["test"],
                *self.dataset.cmapss[fd_target].data["test"],
            )
            source_train_dataset = TensorDataset(*combined_data)
            self._assert_datasets_equal(test_data, source_train_dataset)

    def _assert_datasets_equal(self, baseline_dataset, inner_dataset):
        num_samples = len(baseline_dataset)
        baseline_data = baseline_dataset[:num_samples]
        inner_data = inner_dataset[:num_samples]
        for baseline, inner in zip(baseline_data, inner_data):
            self.assertEqual(0, torch.sum(baseline - inner))

    def test_fail_runs_passed_correctly(self):
        for i in range(1, 5):
            self.assertEqual(0.8, self.dataset.cmapss[i].percent_fail_runs)


class TestPretrainingBaselineDataModuleFullData(
    unittest.TestCase, PretrainingDataModuleTemplate
):
    def setUp(self):
        self.dataset = datasets.PretrainingBaselineDataModule(
            3, num_samples=10000, batch_size=16
        )
        self.dataset.prepare_data()
        self.dataset.setup()

        self.expected_num_val_loaders = 2
        self.window_size = datasets.cmapss.CMAPSSDataModule.WINDOW_SIZES[
            self.dataset.fd_source
        ]

    def test_val_truncation(self):
        with self.subTest(truncation=False):
            dataset = datasets.PretrainingBaselineDataModule(
                3, num_samples=10000, batch_size=16
            )
            self.assertFalse(dataset.source.truncate_val)

        with self.subTest(truncation=True):
            dataset = datasets.PretrainingBaselineDataModule(
                3, num_samples=10000, batch_size=16, truncate_val=True
            )
            self.assertTrue(dataset.source.truncate_val)

    def test_override_window_size(self):
        dataset = datasets.PretrainingBaselineDataModule(
            3, num_samples=10000, batch_size=16, window_size=40
        )
        dataset.prepare_data()
        dataset.setup()
        train_loader = dataset.train_dataloader()

        anchors, queries, _, _ = next(iter(train_loader))
        self.assertEqual(40, anchors.shape[2])
        self.assertEqual(40, queries.shape[2])


class TestPretrainingBaselineDataModuleLowData(
    unittest.TestCase, PretrainingDataModuleTemplate
):
    def setUp(self):
        self.dataset = datasets.PretrainingBaselineDataModule(
            3, percent_broken=0.2, num_samples=10000, batch_size=16
        )
        self.dataset.prepare_data()
        self.dataset.setup()

        self.expected_num_val_loaders = 2
        self.window_size = datasets.cmapss.CMAPSSDataModule.WINDOW_SIZES[
            self.dataset.fd_source
        ]
