import unittest
from unittest import mock

import torch

import datasets
from datasets import cmapss


class TestCMAPSS(unittest.TestCase):
    def test_data(self):
        window_sizes = [30, 20, 30, 15]
        for n, win in enumerate(window_sizes, start=1):
            dataset = cmapss.CMAPSSDataModule(n, batch_size=16)
            dataset.prepare_data()
            dataset.setup()
            for split in ["dev", "val", "test"]:
                with self.subTest(fd=n, split=split):
                    features, targets = dataset.data[split]
                    self.assertEqual(win, features.shape[2])
                    self.assertEqual(len(features), len(targets))
                    self.assertEqual(torch.float32, features.dtype)
                    self.assertEqual(torch.float32, targets.dtype)

    def test_override_window_size(self):
        window_size = 40
        for n in range(1, 5):
            dataset = cmapss.CMAPSSDataModule(n, batch_size=16, window_size=window_size)
            self.assertEqual(window_size, dataset.window_size)
            self.assertEqual(window_size, dataset._loader.window_size)

    def test_default_window_size(self):
        window_sizes = [30, 20, 30, 15]
        for n, win in enumerate(window_sizes, start=1):
            dataset = cmapss.CMAPSSDataModule(n, batch_size=16)
            self.assertEqual(win, dataset.window_size)
            self.assertEqual(win, dataset._loader.window_size)

    def test_feature_select(self):
        feature_idx = [4, 9, 10, 13, 14, 15, 22]
        dataset = cmapss.CMAPSSDataModule(1, batch_size=16, feature_select=feature_idx)
        self.assertListEqual(feature_idx, dataset.feature_select)
        self.assertListEqual(feature_idx, dataset._loader.feature_select)

    def test_default_feature_select(self):
        dataset = cmapss.CMAPSSDataModule(1, batch_size=16)
        self.assertListEqual(dataset._loader.DEFAULT_CHANNELS, dataset.feature_select)
        self.assertListEqual(
            dataset._loader.DEFAULT_CHANNELS, dataset._loader.feature_select
        )

    def test_truncation_functions(self):
        full_dataset = cmapss.CMAPSSDataModule(fd=1, window_size=30, batch_size=4)
        full_dataset.prepare_data()
        full_dataset.setup()

        dataset = cmapss.CMAPSSDataModule(
            fd=1, window_size=30, batch_size=4, percent_fail_runs=0.8
        )
        dataset.prepare_data()
        dataset.setup()
        self.assertGreater(len(full_dataset.data["dev"][0]), len(dataset.data["dev"][0]))
        self.assertEqual(len(full_dataset.data["val"][0]), len(dataset.data["val"][0]))
        self.assertEqual(len(full_dataset.data["test"][0]), len(dataset.data["test"][0]))

        dataset = cmapss.CMAPSSDataModule(
            fd=1, window_size=30, batch_size=4, percent_broken=0.2
        )
        dataset.prepare_data()
        dataset.setup()
        self.assertGreater(len(full_dataset.data["dev"][0]), len(dataset.data["dev"][0]))
        self.assertAlmostEqual(
            0.2,
            len(dataset.data["dev"][0]) / len(full_dataset.data["dev"][0]),
            delta=0.01,
        )
        self.assertEqual(
            len(full_dataset.data["val"][0]), len(dataset.data["val"][0])
        )  # Val dataset_tests not truncated
        self.assertEqual(
            len(full_dataset.data["test"][0]), len(dataset.data["test"][0])
        )  # Test dataset_tests not truncated
        self.assertFalse(
            torch.any(dataset.data["dev"][1] == 1)
        )  # No failure dataset_tests in truncated dataset_tests
        self.assertEqual(
            full_dataset.data["dev"][1][0], dataset.data["dev"][1][0]
        )  # First target has to be equal

    def test_lengths(self):
        for i in range(1, 5):
            dataset = cmapss.CMAPSSDataModule(fd=i, window_size=30, batch_size=4)
            dataset.prepare_data()
            dataset.setup()
            for split in ["dev", "val", "test"]:
                with self.subTest(fd=i, split=split):
                    raw_data, _ = dataset._loader.load_split(split)
                    expected_lenghts = [len(f) for f in raw_data]
                    self.assertListEqual(expected_lenghts, dataset.lengths[split])
                    self.assertEqual(
                        len(dataset.data[split][0]), sum(dataset.lengths[split])
                    )


class DummyCMAPSS:
    def __init__(self, length):
        self.window_size = 30
        self.max_rul = 125
        self.lengths = {"dev": [length]}
        self.data = {
            "dev": (
                torch.zeros(length, self.window_size, 5),
                torch.clamp_max(torch.arange(length, 0, step=-1), 125),
            ),
        }


class DummyCMAPSSShortRuns:
    """Contains runs that are too short with zero features to distinguish them."""

    def __init__(self):
        self.window_size = 30
        self.max_rul = 125
        self.lengths = {"dev": [100, 2, 100, 1, 0]}
        self.data = {
            "dev": (
                torch.cat(
                    [
                        torch.ones(100, self.window_size, 5),  # normal run
                        torch.zeros(2, self.window_size, 5),  # too short run
                        torch.ones(100, self.window_size, 5),  # normal run
                        torch.zeros(1, self.window_size, 5),  # empty run
                    ]
                ),
                torch.cat(
                    [
                        torch.clamp_max(torch.arange(100, 0, step=-1), 125),
                        torch.ones(2) * 500,
                        torch.clamp_max(torch.arange(100, 0, step=-1), 125),
                        torch.ones(1) * 500,
                    ]
                ),
            ),
        }


class TestPairedDataset(unittest.TestCase):
    def setUp(self):
        self.cmapss_normal = DummyCMAPSS(300)
        self.cmapss_short = DummyCMAPSSShortRuns()

    def test_get_pair_idx(self):
        data = datasets.cmapss.PairedCMAPSS([self.cmapss_normal], "dev", 512, 1, True)
        middle_idx = self.cmapss_normal.lengths["dev"][0] // 2
        for _ in range(512):
            anchor_idx, query_idx, _, distance = data._get_pair_idx()
            if anchor_idx < middle_idx:
                self.assertEqual(0, distance)
            else:
                self.assertLessEqual(0, distance)

    def test_get_labeled_pair_idx(self):
        data = datasets.cmapss.PairedCMAPSS([self.cmapss_normal], "dev", 512, 1, True)
        for _ in range(512):
            anchor_idx, query_idx, _, distance = data._get_labeled_pair_idx()
            expected_distance = data._labels[anchor_idx] - data._labels[query_idx]
            self.assertLessEqual(0, distance)
            self.assertEqual(expected_distance, distance)

    def test_pair_func_selection(self):
        with self.subTest("default"):
            data = datasets.cmapss.PairedCMAPSS([self.cmapss_normal], "dev", 512, 1, True)
            self.assertEqual(data._get_pair_idx, data._get_pair_func)
        with self.subTest("False"):
            data = datasets.cmapss.PairedCMAPSS(
                [self.cmapss_normal], "dev", 512, 1, True, labeled=False
            )
            self.assertEqual(data._get_pair_idx, data._get_pair_func)
        with self.subTest("True"):
            data = datasets.cmapss.PairedCMAPSS(
                [self.cmapss_normal], "dev", 512, 1, True, labeled=True
            )
            self.assertEqual(data._get_labeled_pair_idx, data._get_pair_func)

    def test_discarding_too_short_runs(self):
        data = datasets.cmapss.PairedCMAPSS([self.cmapss_short], "dev", 512, 2)
        self.assertTrue((data._features == 1).all())
        self.assertTrue((data._labels < 500).all())
        self.assertListEqual([0, 100, 200], data._run_start_idx.tolist())
        self.assertListEqual([0, 1], data._run_idx.tolist())
        self.assertListEqual([0, 0], data._run_domain_idx.tolist())
