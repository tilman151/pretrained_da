import unittest

import torch
import torch.utils.data
from torch.utils.data import TensorDataset

import datasets


class TestCMAPSS(unittest.TestCase):

    def test__data(self):
        for n, win in enumerate([30, 20, 30, 15], start=1):
            dataset = datasets.CMAPSSDataModule(n, batch_size=16, window_size=win)
            dataset.prepare_data()
            dataset.setup()
            for split in ['dev', 'val', 'test']:
                with self.subTest(fd=n, win=win, split=split):
                    features, targets = dataset.data[split]
                    self.assertEqual(len(features), len(targets))
                    self.assertEqual(torch.float32, features.dtype)
                    self.assertEqual(torch.float32, targets.dtype)

    def test_feature_select(self):
        dataset = datasets.CMAPSSDataModule(1, batch_size=16, window_size=30, feature_select=[4, 9, 10, 13, 14, 15, 22])
        dataset.prepare_data()
        dataset.setup()
        for split in ['dev', 'val', 'test']:
            features, _ = dataset.data[split]
            self.assertEqual(7, features.shape[1])

    def test_truncation_functions(self):
        full_dataset = datasets.CMAPSSDataModule(fd=1, window_size=30, batch_size=4)
        full_dataset.prepare_data()
        full_dataset.setup()

        dataset = datasets.CMAPSSDataModule(fd=1, window_size=30, batch_size=4, percent_fail_runs=0.8)
        dataset.prepare_data()
        dataset.setup()
        self.assertGreater(len(full_dataset.data['dev'][0]), len(dataset.data['dev'][0]))
        self.assertEqual(len(full_dataset.data['val'][0]), len(dataset.data['val'][0]))
        self.assertEqual(len(full_dataset.data['test'][0]), len(dataset.data['test'][0]))

        dataset = datasets.CMAPSSDataModule(fd=1, window_size=30, batch_size=4, percent_broken=0.2)
        dataset.prepare_data()
        dataset.setup()
        self.assertGreater(len(full_dataset.data['dev'][0]), len(dataset.data['dev'][0]))
        self.assertAlmostEqual(0.2, len(dataset.data['dev'][0]) / len(full_dataset.data['dev'][0]), delta=0.01)
        self.assertEqual(len(full_dataset.data['val'][0]), len(dataset.data['val'][0]))  # Val data not truncated
        self.assertEqual(len(full_dataset.data['test'][0]), len(dataset.data['test'][0]))  # Test data not truncated
        self.assertFalse(torch.any(dataset.data['dev'][1] == 1))  # No failure data in truncated data
        self.assertEqual(full_dataset.data['dev'][1][0], dataset.data['dev'][1][0])  # First target has to be equal

    def test_precent_broken_truncation(self):
        full_dataset = datasets.CMAPSSDataModule(fd=1, window_size=30, batch_size=4)
        full_dataset.prepare_data()
        full_dataset.setup()

        truncated_dataset = datasets.CMAPSSDataModule(fd=1, window_size=30, batch_size=4, percent_broken=0.8)
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
                full_dataset = datasets.CMAPSSDataModule(fd=i, window_size=30, batch_size=4)
                full_dataset.prepare_data()
                full_dataset.setup()
                self.assertAlmostEqual(torch.max(full_dataset.data['dev'][0]).item(), 1.)
                self.assertAlmostEqual(torch.min(full_dataset.data['dev'][0]).item(), -1.)

                truncated_dataset = datasets.CMAPSSDataModule(fd=i, window_size=30, batch_size=4, percent_fail_runs=0.8)
                truncated_dataset.prepare_data()
                truncated_dataset.setup()
                self.assertLessEqual(torch.max(truncated_dataset.data['dev'][0]).item(), 1.)
                self.assertGreaterEqual(torch.min(truncated_dataset.data['dev'][0]).item(), -1.)
                self.assertTrue(torch.all(truncated_dataset.data['test'][0] == full_dataset.data['test'][0]))
                self.assertTrue(torch.all(truncated_dataset.data['dev'][0][:50] == full_dataset.data['dev'][0][:50]))

                truncated_dataset = datasets.CMAPSSDataModule(fd=i, window_size=30, batch_size=4, percent_broken=0.2)
                truncated_dataset.prepare_data()
                truncated_dataset.setup()
                self.assertLessEqual(torch.max(truncated_dataset.data['dev'][0]).item(), 1.)
                self.assertGreaterEqual(torch.min(truncated_dataset.data['dev'][0]).item(), -1.)
                self.assertTrue(torch.all(truncated_dataset.data['test'][0] == full_dataset.data['test'][0]))
                self.assertTrue(torch.all(truncated_dataset.data['val'][0] == full_dataset.data['val'][0]))


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

    def test_val_length_equal(self):
        val_loader = self.dataset.val_dataloader()
        source_length = len(self.dataset.source.val_dataloader())
        target_length = len(self.dataset.target.val_dataloader())
        self.assertEqual(max(source_length, target_length), len(val_loader))

    def test_test_length_equal(self):
        test_loader = self.dataset.test_dataloader()
        source_length = len(self.dataset.source.test_dataloader())
        target_length = len(self.dataset.target.test_dataloader())
        self.assertEqual(max(source_length, target_length), len(test_loader))

    def test_train_batch_structure(self):
        train_loader = self.dataset.train_dataloader()
        batch = next(iter(train_loader))
        self.assertEqual(3, len(batch))
        source, source_labels, target = batch
        self.assertEqual(torch.Size((16, 14, 30)), source.shape)
        self.assertEqual(torch.Size((16, 14, 30)), target.shape)
        self.assertEqual(torch.Size((16,)), source_labels.shape)

    def test_val_batch_structure(self):
        val_loader = self.dataset.val_dataloader()
        self._assert_val_test_batch_structure(val_loader)

    def test_test_batch_structure(self):
        test_loader = self.dataset.test_dataloader()
        self._assert_val_test_batch_structure(test_loader)

    def _assert_val_test_batch_structure(self, loader):
        batch = next(iter(loader))
        self.assertEqual(4, len(batch))
        source, source_labels, target, target_labels = batch
        self.assertEqual(torch.Size((16, 14, 30)), source.shape)
        self.assertEqual(torch.Size((16, 14, 30)), target.shape)
        self.assertEqual(torch.Size((16,)), source_labels.shape)
        self.assertEqual(torch.Size((16,)), target_labels.shape)


class TestCMAPSSBaseline(unittest.TestCase):
    def setUp(self):
        self.dataset = datasets.BaselineDataModule(3, 1, batch_size=16, window_size=30)
        self.dataset.prepare_data()
        self.dataset.setup()

    def test_train_batch_structure(self):
        train_loader = self.dataset.train_dataloader()
        self._assert_train_val_batch_structure(train_loader)

    def test_val_batch_structure(self):
        val_loader = self.dataset.val_dataloader()
        self._assert_train_val_batch_structure(val_loader)

    def test_test_batch_structure(self):
        test_loader = self.dataset.test_dataloader()
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
        baseline_train_dataset = self.dataset._to_dataset('dev')
        source_train_dataset = self.dataset.source._to_dataset(*self.dataset.source.data['dev'])
        self._assert_datasets_equal(baseline_train_dataset, source_train_dataset)

    def test_selected_source_on_val(self):
        baseline_train_dataset = self.dataset._to_dataset('val')
        source_train_dataset = self.dataset.source._to_dataset(*self.dataset.source.data['val'])
        self._assert_datasets_equal(baseline_train_dataset, source_train_dataset)

    def test_selected_both_on_test(self):
        baseline_train_dataset = self.dataset._to_dataset('test')
        combined_data = datasets._unify_source_and_target_length(*self.dataset.source.data['test'],
                                                                 *self.dataset.target.data['test'])
        source_train_dataset = TensorDataset(*combined_data)
        self._assert_datasets_equal(baseline_train_dataset, source_train_dataset)

    def _assert_datasets_equal(self, baseline_dataset, inner_dataset):
        num_samples = len(baseline_dataset)
        baseline_data = baseline_dataset[:num_samples]
        inner_data = inner_dataset[:num_samples]
        for baseline, inner in zip(baseline_data, inner_data):
            self.assertEqual(0, torch.sum(baseline - inner))
