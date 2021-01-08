import unittest

import hyperopt.hyperopt_transfer as hyperopt_transfer


class TestTransferHyperopt(unittest.TestCase):
    def test_tune_function(self):
        config = {
            "num_layers": 8,
            "base_filters": 16,
            "domain_tradeoff": 1.0,
            "latent_dim": 32,
            "dropout": 0.1,
            "num_disc_layers": 1,
            "lr": 0.01,
            "batch_size": 512,
        }
        hyperopt_transfer.tune_transfer(config, 2, 1, 0.8)
