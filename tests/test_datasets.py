import unittest
from unittest import mock

import numpy as np
import torch
import torch.utils.data
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler

import datasets
from datasets import cmapss
from datasets.adaption import _unify_source_and_target_length
from datasets.cmapss import PairedCMAPSS


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
            dataset.prepare_data()
            dataset.setup()
            for split in ["dev", "val", "test"]:
                with self.subTest(fd=n, split=split):
                    features, targets = dataset.data[split]
                    self.assertEqual(window_size, features.shape[2])

    def test_feature_select(self):
        dataset = cmapss.CMAPSSDataModule(
            1, batch_size=16, window_size=30, feature_select=[4, 9, 10, 13, 14, 15, 22]
        )
        dataset.prepare_data()
        dataset.setup()
        for split in ["dev", "val", "test"]:
            features, _ = dataset.data[split]
            self.assertEqual(7, features.shape[1])

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
        )  # Val data not truncated
        self.assertEqual(
            len(full_dataset.data["test"][0]), len(dataset.data["test"][0])
        )  # Test data not truncated
        self.assertFalse(
            torch.any(dataset.data["dev"][1] == 1)
        )  # No failure data in truncated data
        self.assertEqual(
            full_dataset.data["dev"][1][0], dataset.data["dev"][1][0]
        )  # First target has to be equal

    def test_precent_broken_truncation(self):
        full_dataset = cmapss.CMAPSSDataModule(fd=1, window_size=30, batch_size=4)
        full_dataset.prepare_data()
        full_dataset.setup()

        truncated_dataset = cmapss.CMAPSSDataModule(
            fd=1, window_size=30, batch_size=4, percent_broken=0.8
        )
        truncated_dataset.prepare_data()
        truncated_dataset.setup()

        features = [torch.randn(n, 30) for n in torch.randint(50, 200, (100,))]
        truncated_features = truncated_dataset._truncate_features(
            features.copy()
        )  # pass copy to get a new list

        for n, (full_feat, trunc_feat) in enumerate(zip(features, truncated_features)):
            with self.subTest(n=n):
                last_idx = trunc_feat.shape[0]
                self.assertTrue(torch.all(full_feat[:last_idx] == trunc_feat))

    def test_normalization_min_max(self):
        for i in range(1, 5):
            with self.subTest(fd=i):
                full_dataset = cmapss.CMAPSSDataModule(fd=i, window_size=30, batch_size=4)
                full_dataset.prepare_data()
                full_dataset.setup()
                self.assertAlmostEqual(torch.max(full_dataset.data["dev"][0]).item(), 1.0)
                self.assertAlmostEqual(
                    torch.min(full_dataset.data["dev"][0]).item(), -1.0
                )

                truncated_dataset = cmapss.CMAPSSDataModule(
                    fd=i, window_size=30, batch_size=4, percent_fail_runs=0.8
                )
                truncated_dataset.prepare_data()
                truncated_dataset.setup()
                self.assertLessEqual(
                    torch.max(truncated_dataset.data["dev"][0]).item(), 1.0
                )
                self.assertGreaterEqual(
                    torch.min(truncated_dataset.data["dev"][0]).item(), -1.0
                )
                self.assertTrue(
                    torch.all(
                        truncated_dataset.data["test"][0] == full_dataset.data["test"][0]
                    )
                )
                self.assertTrue(
                    torch.all(
                        truncated_dataset.data["dev"][0][:50]
                        == full_dataset.data["dev"][0][:50]
                    )
                )

                truncated_dataset = cmapss.CMAPSSDataModule(
                    fd=i, window_size=30, batch_size=4, percent_broken=0.2
                )
                truncated_dataset.prepare_data()
                truncated_dataset.setup()
                self.assertLessEqual(
                    torch.max(truncated_dataset.data["dev"][0]).item(), 1.0
                )
                self.assertGreaterEqual(
                    torch.min(truncated_dataset.data["dev"][0]).item(), -1.0
                )
                self.assertTrue(
                    torch.all(
                        truncated_dataset.data["test"][0] == full_dataset.data["test"][0]
                    )
                )
                self.assertTrue(
                    torch.all(
                        truncated_dataset.data["val"][0] == full_dataset.data["val"][0]
                    )
                )

    def test_lengths(self):
        for i in range(1, 5):
            dataset = cmapss.CMAPSSDataModule(fd=i, window_size=30, batch_size=4)
            dataset.prepare_data()
            dataset.setup()
            for split in ["dev", "val", "test"]:
                with self.subTest(fd=i, split=split):
                    raw_data = dataset._load_features(dataset._file_path(split))
                    # lengths should be length of raw data minus one less than window_size
                    expected_lenghts = (
                        [1] * len(raw_data)
                        if split == "test"
                        else [len(f) for f in raw_data]
                    )
                    self.assertListEqual(expected_lenghts, dataset.lengths[split])
                    self.assertEqual(
                        len(dataset.data[split][0]), sum(dataset.lengths[split])
                    )

    @mock.patch("datasets.cmapss.CMAPSSDataModule._truncate_features", wraps=lambda x: x)
    def test_val_truncation(self, mock_truncate):
        dataset = cmapss.CMAPSSDataModule(fd=1, window_size=30, batch_size=4)
        dataset.prepare_data()
        with self.subTest(truncate_val=False):
            dataset._setup_split("dev")
            mock_truncate.assert_called_once()
            mock_truncate.reset_mock()
            dataset._setup_split("val")
            mock_truncate.assert_not_called()

        dataset = cmapss.CMAPSSDataModule(
            fd=1, window_size=30, batch_size=4, truncate_val=True
        )
        dataset.prepare_data()
        with self.subTest(truncate_val=True):
            dataset._setup_split("dev")
            mock_truncate.assert_called_once()
            mock_truncate.reset_mock()
            dataset._setup_split("val")
            mock_truncate.assert_called_once()


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


class TestAdaptionDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = datasets.adaption.AdaptionDataset(
            torch.arange(100), torch.arange(100), torch.arange(150)
        )

    def test_source_target_shuffeled(self):
        for i in range(len(self.dataset)):
            source_one, label_one, target_one = self.dataset[i]
            source_another, label_another, target_another = self.dataset[i]
            self.assertEqual(source_one, source_another)
            self.assertEqual(label_one, label_another)
            self.assertNotEqual(target_one, target_another)


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
