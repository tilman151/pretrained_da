import unittest

import torch

import datasets
import datasets.cmapss
from tests.data.templates import PretrainingDataModuleTemplate


class TestCMAPSSAdaption(unittest.TestCase):
    def setUp(self):
        self.dataset = datasets.DomainAdaptionDataModule(3, 2, batch_size=16)
        self.dataset.prepare_data()
        self.dataset.setup()

    def test_window_size(self):
        with self.subTest(case="bigger2smaller"):
            train_loader = self.dataset.train_dataloader()
            source, _, target = next(iter(train_loader))
            self.assertEqual(
                datasets.cmapss.CMAPSSDataModule.WINDOW_SIZES[2], source.shape[2]
            )
            self.assertEqual(
                datasets.cmapss.CMAPSSDataModule.WINDOW_SIZES[2], target.shape[2]
            )

        with self.subTest(case="smaller2bigger"):
            dataset = datasets.DomainAdaptionDataModule(2, 3, batch_size=16)
            dataset.prepare_data()
            dataset.setup()
            train_loader = dataset.train_dataloader()
            source, _, target = next(iter(train_loader))
            self.assertEqual(
                datasets.cmapss.CMAPSSDataModule.WINDOW_SIZES[3], source.shape[2]
            )
            self.assertEqual(
                datasets.cmapss.CMAPSSDataModule.WINDOW_SIZES[3], target.shape[2]
            )

    def test_override_window_size(self):
        dataset = datasets.DomainAdaptionDataModule(3, 2, batch_size=16, window_size=40)
        dataset.prepare_data()
        dataset.setup()
        train_loader = dataset.train_dataloader()

        source, _, target = next(iter(train_loader))
        self.assertEqual(40, source.shape[2])
        self.assertEqual(40, target.shape[2])

    def test_val_source_target_order(self):
        val_source_loader, val_target_loader, _ = self.dataset.val_dataloader()
        self._assert_datasets_equal(
            val_source_loader.dataset,
            self.dataset.source._to_dataset(*self.dataset.source.data["val"]),
        )
        self._assert_datasets_equal(
            val_target_loader.dataset,
            self.dataset.source._to_dataset(*self.dataset.target.data["val"]),
        )

    def test_test_source_target_order(self):
        test_source_loader, test_target_loader = self.dataset.test_dataloader()
        self._assert_datasets_equal(
            test_source_loader.dataset,
            self.dataset.source._to_dataset(*self.dataset.source.data["test"]),
        )
        self._assert_datasets_equal(
            test_target_loader.dataset,
            self.dataset.source._to_dataset(*self.dataset.target.data["test"]),
        )

    def _assert_datasets_equal(self, adaption_dataset, inner_dataset):
        num_samples = len(adaption_dataset)
        baseline_data = adaption_dataset[:num_samples]
        inner_data = inner_dataset[:num_samples]
        for baseline, inner in zip(baseline_data, inner_data):
            self.assertEqual(0, torch.sum(baseline - inner))

    def test_train_batch_structure(self):
        window_size = datasets.cmapss.CMAPSSDataModule.WINDOW_SIZES[
            self.dataset.fd_target
        ]
        train_loader = self.dataset.train_dataloader()
        batch = next(iter(train_loader))
        self.assertEqual(3, len(batch))
        source, source_labels, target = batch
        self.assertEqual(torch.Size((16, 14, window_size)), source.shape)
        self.assertEqual(torch.Size((16, 14, window_size)), target.shape)
        self.assertEqual(torch.Size((16,)), source_labels.shape)

    def test_val_batch_structure(self):
        val_source_loader, val_target_loader, _ = self.dataset.val_dataloader()
        self._assert_val_test_batch_structure(val_source_loader)
        self._assert_val_test_batch_structure(val_target_loader)

    def test_test_batch_structure(self):
        test_source_loader, test_target_loader = self.dataset.test_dataloader()
        self._assert_val_test_batch_structure(test_source_loader)
        self._assert_val_test_batch_structure(test_target_loader)

    def _assert_val_test_batch_structure(self, loader):
        window_size = datasets.cmapss.CMAPSSDataModule.WINDOW_SIZES[
            self.dataset.fd_target
        ]
        batch = next(iter(loader))
        self.assertEqual(2, len(batch))
        features, labels = batch
        self.assertEqual(torch.Size((16, 14, window_size)), features.shape)
        self.assertEqual(torch.Size((16,)), labels.shape)


class TestPretrainingDataModuleFullData(unittest.TestCase, PretrainingDataModuleTemplate):
    def setUp(self):
        self.dataset = datasets.PretrainingAdaptionDataModule(
            3, 2, num_samples=10000, batch_size=16
        )
        self.dataset.prepare_data()
        self.dataset.setup()

        self.expected_num_val_loaders = 3
        self.window_size = datasets.cmapss.CMAPSSDataModule.WINDOW_SIZES[
            self.dataset.fd_target
        ]

    def test_target_val_truncation(self):
        with self.subTest(truncation=False):
            dataset = datasets.PretrainingAdaptionDataModule(
                3, 2, num_samples=10000, batch_size=16
            )
            self.assertFalse(dataset.target.truncate_val)

        with self.subTest(truncation=True):
            dataset = datasets.PretrainingAdaptionDataModule(
                3, 2, num_samples=10000, batch_size=16, truncate_target_val=True
            )
            self.assertTrue(dataset.target.truncate_val)

    def test_override_window_size(self):
        dataset = datasets.PretrainingAdaptionDataModule(
            3, 2, num_samples=10000, batch_size=16, window_size=40
        )
        dataset.prepare_data()
        dataset.setup()
        train_loader = dataset.train_dataloader()

        anchors, queries, _, _ = next(iter(train_loader))
        self.assertEqual(40, anchors.shape[2])
        self.assertEqual(40, queries.shape[2])


class TestPretrainingDataModuleLowData(unittest.TestCase, PretrainingDataModuleTemplate):
    def setUp(self):
        self.dataset = datasets.PretrainingAdaptionDataModule(
            1, 3, percent_broken=0.2, num_samples=10000, batch_size=16
        )
        self.dataset.prepare_data()
        self.dataset.setup()

        self.expected_num_val_loaders = 3
        self.window_size = datasets.cmapss.CMAPSSDataModule.WINDOW_SIZES[
            self.dataset.fd_target
        ]
