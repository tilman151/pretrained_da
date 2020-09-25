import unittest
import torch
import torch.utils.data

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
        self.assertEqual(len(full_dataset.data['test'][0]), len(dataset.data['test'][0]))

        dataset = datasets.CMAPSSDataModule(fd=1, window_size=30, batch_size=4, percent_broken=0.2)
        dataset.prepare_data()
        dataset.setup()
        self.assertGreater(len(full_dataset.data['dev'][0]), len(dataset.data['dev'][0]))
        self.assertEqual(len(full_dataset.data['test'][0]), len(dataset.data['test'][0]))
        self.assertFalse(torch.any(dataset.data['dev'][1] == 1))

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
