import unittest

import numpy as np
import torch
import torch.utils.data
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler

import datasets
from datasets import cmapss
from datasets.adaption import _unify_source_and_target_length
from datasets.cmapss import PairedCMAPSS


class TestCMAPSS(unittest.TestCase):

    def test__data(self):
        for n, win in enumerate([30, 20, 30, 15], start=1):
            dataset = cmapss.CMAPSSDataModule(n, batch_size=16, window_size=win)
            dataset.prepare_data()
            dataset.setup()
            for split in ['dev', 'val', 'test']:
                with self.subTest(fd=n, win=win, split=split):
                    features, targets = dataset.data[split]
                    self.assertEqual(len(features), len(targets))
                    self.assertEqual(torch.float32, features.dtype)
                    self.assertEqual(torch.float32, targets.dtype)

    def test_feature_select(self):
        dataset = cmapss.CMAPSSDataModule(1, batch_size=16, window_size=30, feature_select=[4, 9, 10, 13, 14, 15, 22])
        dataset.prepare_data()
        dataset.setup()
        for split in ['dev', 'val', 'test']:
            features, _ = dataset.data[split]
            self.assertEqual(7, features.shape[1])

    def test_truncation_functions(self):
        full_dataset = cmapss.CMAPSSDataModule(fd=1, window_size=30, batch_size=4)
        full_dataset.prepare_data()
        full_dataset.setup()

        dataset = cmapss.CMAPSSDataModule(fd=1, window_size=30, batch_size=4, percent_fail_runs=0.8)
        dataset.prepare_data()
        dataset.setup()
        self.assertGreater(len(full_dataset.data['dev'][0]), len(dataset.data['dev'][0]))
        self.assertEqual(len(full_dataset.data['val'][0]), len(dataset.data['val'][0]))
        self.assertEqual(len(full_dataset.data['test'][0]), len(dataset.data['test'][0]))

        dataset = cmapss.CMAPSSDataModule(fd=1, window_size=30, batch_size=4, percent_broken=0.2)
        dataset.prepare_data()
        dataset.setup()
        self.assertGreater(len(full_dataset.data['dev'][0]), len(dataset.data['dev'][0]))
        self.assertAlmostEqual(0.2, len(dataset.data['dev'][0]) / len(full_dataset.data['dev'][0]), delta=0.01)
        self.assertEqual(len(full_dataset.data['val'][0]), len(dataset.data['val'][0]))  # Val data not truncated
        self.assertEqual(len(full_dataset.data['test'][0]), len(dataset.data['test'][0]))  # Test data not truncated
        self.assertFalse(torch.any(dataset.data['dev'][1] == 1))  # No failure data in truncated data
        self.assertEqual(full_dataset.data['dev'][1][0], dataset.data['dev'][1][0])  # First target has to be equal

    def test_precent_broken_truncation(self):
        full_dataset = cmapss.CMAPSSDataModule(fd=1, window_size=30, batch_size=4)
        full_dataset.prepare_data()
        full_dataset.setup()

        truncated_dataset = cmapss.CMAPSSDataModule(fd=1, window_size=30, batch_size=4, percent_broken=0.8)
        truncated_dataset.prepare_data()
        truncated_dataset.setup()

        features = [torch.randn(n, 30) for n in torch.randint(50, 200, (100,))]
        truncated_features = truncated_dataset._truncate_features(features.copy())  # pass copy to get a new list

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
                self.assertAlmostEqual(torch.max(full_dataset.data['dev'][0]).item(), 1.)
                self.assertAlmostEqual(torch.min(full_dataset.data['dev'][0]).item(), -1.)

                truncated_dataset = cmapss.CMAPSSDataModule(fd=i, window_size=30, batch_size=4, percent_fail_runs=0.8)
                truncated_dataset.prepare_data()
                truncated_dataset.setup()
                self.assertLessEqual(torch.max(truncated_dataset.data['dev'][0]).item(), 1.)
                self.assertGreaterEqual(torch.min(truncated_dataset.data['dev'][0]).item(), -1.)
                self.assertTrue(torch.all(truncated_dataset.data['test'][0] == full_dataset.data['test'][0]))
                self.assertTrue(torch.all(truncated_dataset.data['dev'][0][:50] == full_dataset.data['dev'][0][:50]))

                truncated_dataset = cmapss.CMAPSSDataModule(fd=i, window_size=30, batch_size=4, percent_broken=0.2)
                truncated_dataset.prepare_data()
                truncated_dataset.setup()
                self.assertLessEqual(torch.max(truncated_dataset.data['dev'][0]).item(), 1.)
                self.assertGreaterEqual(torch.min(truncated_dataset.data['dev'][0]).item(), -1.)
                self.assertTrue(torch.all(truncated_dataset.data['test'][0] == full_dataset.data['test'][0]))
                self.assertTrue(torch.all(truncated_dataset.data['val'][0] == full_dataset.data['val'][0]))

    def test_lengths(self):
        for i in range(1, 5):
            dataset = cmapss.CMAPSSDataModule(fd=i, window_size=30, batch_size=4)
            dataset.prepare_data()
            dataset.setup()
            for split in ['dev', 'val', 'test']:
                with self.subTest(fd=i, split=split):
                    raw_data = dataset._load_features(dataset._file_path(split))
                    # lengths should be length of raw data minus one less than window_size
                    expected_lenghts = [1] * len(raw_data) if split == 'test' else [len(f) - 29 for f in raw_data]
                    self.assertListEqual(expected_lenghts, dataset.lengths[split])


class TestCMAPSSAdaption(unittest.TestCase):
    def setUp(self):
        self.dataset = datasets.DomainAdaptionDataModule(3, 1, batch_size=16, window_size=30)
        self.dataset.prepare_data()
        self.dataset.setup()

    def test_train_length_equal(self):
        train_loader = self.dataset.train_dataloader()
        source_length = len(self.dataset.source.train_dataloader())
        target_length = len(self.dataset.target.train_dataloader())
        self.assertEqual(max(source_length, target_length), len(train_loader))

    def test_val_source_target_order(self):
        val_source_loader, val_target_loader = self.dataset.val_dataloader()
        self._assert_datasets_equal(val_source_loader.dataset,
                                    self.dataset.source._to_dataset(*self.dataset.source.data['val']))
        self._assert_datasets_equal(val_target_loader.dataset,
                                    self.dataset.source._to_dataset(*self.dataset.target.data['val']))

    def test_test_source_target_order(self):
        test_source_loader, test_target_loader = self.dataset.test_dataloader()
        self._assert_datasets_equal(test_source_loader.dataset,
                                    self.dataset.source._to_dataset(*self.dataset.source.data['test']))
        self._assert_datasets_equal(test_target_loader.dataset,
                                    self.dataset.source._to_dataset(*self.dataset.target.data['test']))

    def _assert_datasets_equal(self, adaption_dataset, inner_dataset):
        num_samples = len(adaption_dataset)
        baseline_data = adaption_dataset[:num_samples]
        inner_data = inner_dataset[:num_samples]
        for baseline, inner in zip(baseline_data, inner_data):
            self.assertEqual(0, torch.sum(baseline - inner))

    def test_train_batch_structure(self):
        train_loader = self.dataset.train_dataloader()
        batch = next(iter(train_loader))
        self.assertEqual(3, len(batch))
        source, source_labels, target = batch
        self.assertEqual(torch.Size((16, 14, 30)), source.shape)
        self.assertEqual(torch.Size((16, 14, 30)), target.shape)
        self.assertEqual(torch.Size((16,)), source_labels.shape)

    def test_val_batch_structure(self):
        val_source_loader, val_target_loader = self.dataset.val_dataloader()
        self._assert_val_test_batch_structure(val_source_loader)
        self._assert_val_test_batch_structure(val_target_loader)

    def test_test_batch_structure(self):
        test_source_loader, test_target_loader = self.dataset.test_dataloader()
        self._assert_val_test_batch_structure(test_source_loader)
        self._assert_val_test_batch_structure(test_target_loader)

    def _assert_val_test_batch_structure(self, loader):
        batch = next(iter(loader))
        self.assertEqual(2, len(batch))
        features, labels = batch
        self.assertEqual(torch.Size((16, 14, 30)), features.shape)
        self.assertEqual(torch.Size((16,)), labels.shape)


class TestCMAPSSBaseline(unittest.TestCase):
    def setUp(self):
        self.dataset = datasets.BaselineDataModule(3, batch_size=16, window_size=30)
        self.dataset.prepare_data()
        self.dataset.setup()

    def test_train_batch_structure(self):
        train_loader = self.dataset.train_dataloader()
        self.assertIsInstance(train_loader.sampler, RandomSampler)
        self._assert_train_val_batch_structure(train_loader)

    def test_val_batch_structure(self):
        val_loader = self.dataset.val_dataloader()
        self.assertIsInstance(val_loader.sampler, SequentialSampler)
        self._assert_train_val_batch_structure(val_loader)

    def test_test_batch_structure(self):
        test_loaders = self.dataset.test_dataloader()
        for test_loader in test_loaders:
            self.assertIsInstance(test_loader.sampler, SequentialSampler)
            batch = next(iter(test_loader))
            self.assertEqual(4, len(batch))
            source, source_labels, target, target_labels = batch
            self.assertEqual(torch.Size((16, 14, 30)), source.shape)
            self.assertEqual(torch.Size((16, 14, 30)), target.shape)
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
        baseline_train_dataset = self.dataset._to_dataset(fd_source, split='dev')
        source_train_dataset = self.dataset.cmapss[fd_source]._to_dataset(*self.dataset.cmapss[fd_source].data['dev'])
        self._assert_datasets_equal(baseline_train_dataset, source_train_dataset)

    def test_selected_source_on_val(self):
        fd_source = self.dataset.fd_source
        baseline_train_dataset = self.dataset._to_dataset(fd_source, split='val')
        source_train_dataset = self.dataset.cmapss[fd_source]._to_dataset(*self.dataset.cmapss[fd_source].data['val'])
        self._assert_datasets_equal(baseline_train_dataset, source_train_dataset)

    def test_selected_both_on_test(self):
        fd_source = self.dataset.fd_source
        fd_target = 1
        baseline_train_dataset = self.dataset._to_dataset(fd_source, fd_target, split='test')
        combined_data = _unify_source_and_target_length(*self.dataset.cmapss[fd_source].data['test'],
                                                        *self.dataset.cmapss[fd_target].data['test'])
        source_train_dataset = TensorDataset(*combined_data)
        self._assert_datasets_equal(baseline_train_dataset, source_train_dataset)

    def test_received_all_datasets_on_test(self):
        test_loaders = self.dataset.test_dataloader()
        fd_source = self.dataset.fd_source
        for fd_target, test_loader in enumerate(test_loaders, start=1):
            test_data = test_loader.dataset
            combined_data = _unify_source_and_target_length(*self.dataset.cmapss[fd_source].data['test'],
                                                            *self.dataset.cmapss[fd_target].data['test'])
            source_train_dataset = TensorDataset(*combined_data)
            self._assert_datasets_equal(test_data, source_train_dataset)

    def _assert_datasets_equal(self, baseline_dataset, inner_dataset):
        num_samples = len(baseline_dataset)
        baseline_data = baseline_dataset[:num_samples]
        inner_data = inner_dataset[:num_samples]
        for baseline, inner in zip(baseline_data, inner_data):
            self.assertEqual(0, torch.sum(baseline - inner))


class PretrainingDataModuleTemplate:
    def test_build_pairs(self):
        for split in ['dev', 'val']:
            with self.subTest(split=split):
                paired_dataset = self.dataset._get_paired_dataset(split)
                pairs = self._get_pairs(paired_dataset)
                self.assertTrue(np.all(pairs[:, 0] < pairs[:, 1]))  # query always after anchor
                self.assertTrue(np.all(pairs[:, 2] <= 1))  # domain label is either 1
                self.assertTrue(np.all(pairs[:, 2] >= 0))  # or zero
                run_start_idx = paired_dataset._run_start_idx
                run_idx_of_pair = self._get_run_idx_of_pair(pairs, run_start_idx)
                query_in_same_run = [run_start_idx[run_idx + 1] > query_idx
                                     for run_idx, query_idx in zip(run_idx_of_pair, pairs[:, 1])]
                self.assertTrue(all(query_in_same_run))

    def test_domain_labels(self):
        for split in ['dev', 'val']:
            with self.subTest(split=split):
                paired_dataset = self.dataset._get_paired_dataset(split)
                pairs = self._get_pairs(paired_dataset)
                run_start_idx = paired_dataset._run_start_idx
                run_idx_of_pair = self._get_run_idx_of_pair(pairs, run_start_idx)
                run_domain_idx = paired_dataset._run_domain_idx

                expected_domain_labels = [run_domain_idx[run_idx] for run_idx in run_idx_of_pair]
                actual_domain_labels = pairs[:, 2].tolist()
                self.assertListEqual(expected_domain_labels, actual_domain_labels)

    def _get_run_idx_of_pair(self, pairs, run_start_idx):
        """Run idx is number of start idx smaller than or equal to anchor minus one."""
        return np.sum(run_start_idx[:, None] <= pairs[:, 0], axis=0) - 1

    def test_min_distance(self):
        dataset = datasets.PretrainingAdaptionDataModule(3, 1, num_samples=10000, min_distance=30, batch_size=16, window_size=30)
        dataset.prepare_data()
        dataset.setup()

        for split in ['dev', 'val']:
            with self.subTest(split=split):
                pairs = self._get_pairs(dataset._get_paired_dataset(split))
                distances = pairs[:, 1] - pairs[:, 0]
                self.assertTrue(np.all(pairs[:, 0] < pairs[:, 1]))
                self.assertTrue(np.all(distances >= 30))

    def _get_pairs(self, paired_dataset):
        pairs = np.array([paired_dataset._get_pair_idx() for _ in range(paired_dataset.num_samples)])

        return pairs

    def test_data_structure(self):
        with self.subTest(split='dev'):
            dataloader = self.dataset.train_dataloader()
            self._check_paired_dataset(dataloader.dataset)

        with self.subTest(split='val'):
            loaders = self.dataset.val_dataloader()
            self.assertIsInstance(loaders, list)
            self.assertEqual(3, len(loaders))
            self._check_paired_dataset(loaders[0].dataset)
            for dataloader in loaders[1:]:
                self._check_tensor_dataset(dataloader.dataset)

    def _check_paired_dataset(self, data):
        self.assertIsInstance(data, PairedCMAPSS)
        self._check_paired_shapes(data)

    def _check_paired_shapes(self, data):
        for anchors, queries, distances, domain_labels in data:
            self.assertEqual(torch.Size((14, 30)), anchors.shape)
            self.assertEqual(torch.Size((14, 30)), queries.shape)
            self.assertEqual(torch.Size(()), distances.shape)
            self.assertEqual(torch.Size(()), domain_labels.shape)

    def _check_tensor_dataset(self, data):
        self.assertIsInstance(data, TensorDataset)
        self._check_cmapss_shapes(data)

    def _check_cmapss_shapes(self, data):
        for i in range(len(data)):
            features, labels = data[i]
            self.assertEqual(torch.Size((14, 30)), features.shape)
            self.assertEqual(torch.Size(()), labels.shape)

    def test_distances(self):
        with self.subTest(split='dev'):
            _, _, distances, _ = self._run_epoch(self.dataset.train_dataloader())
            self.assertTrue(torch.all(distances > 0))

        with self.subTest(split='val'):
            _, _, distances, _ = self._run_epoch(self.dataset.val_dataloader()[0])
            self.assertTrue(torch.all(distances > 0))

    def test_determinism(self):
        with self.subTest(split='dev'):
            train_loader = self.dataset.train_dataloader()
            one_train_data = self._run_epoch(train_loader)
            another_train_data = self._run_epoch(train_loader)

            for one, another in zip(one_train_data, another_train_data):
                self.assertNotEqual(0., torch.sum(one - another))

        with self.subTest(split='val'):
            val_loader, _, _ = self.dataset.val_dataloader()
            one_train_data = self._run_epoch(val_loader)
            another_train_data = self._run_epoch(val_loader)

            for one, another in zip(one_train_data, another_train_data):
                self.assertEqual(0., torch.sum(one - another))

    def _run_epoch(self, loader):
        anchors = torch.empty((len(loader.dataset), 14, 30))
        queries = torch.empty((len(loader.dataset), 14, 30))
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
        self.dataset = datasets.PretrainingAdaptionDataModule(3, 1, num_samples=10000, batch_size=16, window_size=30)
        self.dataset.prepare_data()
        self.dataset.setup()


class TestPretrainingDataModuleLowData(unittest.TestCase, PretrainingDataModuleTemplate):
    def setUp(self):
        self.dataset = datasets.PretrainingAdaptionDataModule(3, 1, percent_broken=0.2, num_samples=10000,
                                                              batch_size=16, window_size=30)
        self.dataset.prepare_data()
        self.dataset.setup()
