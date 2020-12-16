import unittest
from unittest import mock

import run_complete
import run_daan


class TestRunDann(unittest.TestCase):
    @mock.patch('pytorch_lightning.callbacks.ModelCheckpoint')
    @mock.patch('pytorch_lightning.Trainer')
    @mock.patch('lightning.logger.MLTBLogger')
    @mock.patch('lightning.daan.DAAN')
    @mock.patch('datasets.DomainAdaptionDataModule')
    def test_run(self, mock_datamodule, mock_dann, mock_logger, mock_trainer, mock_callback):
        source = 3
        target = 1
        percent_broken = 0.6
        domain_tradeoff = 1.0
        record_embeddings = False
        seed = 42
        gpu = 1
        pretrained_encoder_path = None
        with self.subTest(pre_trained=False):
            run_daan.run(source, target, percent_broken, domain_tradeoff,
                         record_embeddings, seed, gpu, pretrained_encoder_path)

            mock_datamodule.assert_called_with(fd_source=source,
                                               fd_target=target,
                                               batch_size=512,
                                               window_size=30,
                                               percent_broken=percent_broken)
            mock_trainer.assert_called_with(gpus=[gpu],
                                            max_epochs=200,
                                            logger=mock_logger(),
                                            deterministic=True,
                                            log_every_n_steps=10,
                                            checkpoint_callback=mock_callback(),
                                            val_check_interval=1.0)
            mock_dann.assert_called_with(in_channels=14,
                                         seq_len=30,
                                         num_layers=4,
                                         kernel_size=3,
                                         base_filters=16,
                                         latent_dim=128,
                                         domain_trade_off=domain_tradeoff,
                                         domain_disc_dim=64,
                                         num_disc_layers=2,
                                         optim_type='adam',
                                         lr=0.01,
                                         record_embeddings=record_embeddings)
            mock_dann.load_encoder.assert_not_called()

        with self.subTest(pre_trained=True):
            pretrained_encoder_path = 'test'
            run_daan.run(source, target, percent_broken, domain_tradeoff,
                         record_embeddings, seed, gpu, pretrained_encoder_path)
            mock_trainer.assert_called_with(gpus=[gpu],
                                            max_epochs=10,
                                            logger=mock_logger(),
                                            deterministic=True,
                                            log_every_n_steps=10,
                                            checkpoint_callback=mock_callback(),
                                            val_check_interval=0.1)
            mock_dann.assert_called_with(in_channels=14,
                                         seq_len=30,
                                         num_layers=4,
                                         kernel_size=3,
                                         base_filters=16,
                                         latent_dim=128,
                                         domain_trade_off=domain_tradeoff,
                                         domain_disc_dim=64,
                                         num_disc_layers=2,
                                         optim_type='adam',
                                         lr=0.01,
                                         record_embeddings=record_embeddings)
            mock_dann().load_encoder.assert_called_with(pretrained_encoder_path, load_disc=True)

    @mock.patch('run_daan.run')
    def test_run_multiple(self, mock_run):
        source = 3
        target = 1
        broken = [0.6, 0.4]
        domain_tradeoff = [1.0, 0.1]
        record_embeddings = False
        gpu = 1
        replications = 2
        pretrained_path = None
        run_daan.run_multiple(source, target, broken, domain_tradeoff,
                              record_embeddings, replications, gpu, pretrained_path)

        calls = [mock.call(source, target, broken[0], domain_tradeoff[0], record_embeddings, 1343278, gpu,
                           pretrained_path),
                 mock.call(source, target, broken[0], domain_tradeoff[0], record_embeddings, 9525887, gpu,
                           pretrained_path),
                 mock.call(source, target, broken[0], domain_tradeoff[1], record_embeddings, 1343278, gpu,
                           pretrained_path),
                 mock.call(source, target, broken[0], domain_tradeoff[1], record_embeddings, 9525887, gpu,
                           pretrained_path),
                 mock.call(source, target, broken[1], domain_tradeoff[0], record_embeddings, 1343278, gpu,
                           pretrained_path),
                 mock.call(source, target, broken[1], domain_tradeoff[0], record_embeddings, 9525887, gpu,
                           pretrained_path),
                 mock.call(source, target, broken[1], domain_tradeoff[1], record_embeddings, 1343278, gpu,
                           pretrained_path),
                 mock.call(source, target, broken[1], domain_tradeoff[1], record_embeddings, 9525887, gpu,
                           pretrained_path),
                 ]

        mock_run.assert_has_calls(calls)


class TestRunComplete(unittest.TestCase):
    @mock.patch('run_complete.run_daan')
    @mock.patch('run_complete.run_pretraining')
    def test_run_with_all_pretrained(self, mock_pretraining, mock_dann):
        source = 3
        target = 1
        broken = [0.6, 0.4]
        domain_tradeoff = [0.1]
        dropout = 0.1
        record_embeddings = False
        pretraining_reps = 2
        best_only = False
        gpu = 1
        high_val_pretrained_path = 'high_val'
        low_val_pretrained_path = 'low_val'
        mock_pretraining.return_value = {broken[0]: {0.01: [(high_val_pretrained_path, 0.5),
                                                            (low_val_pretrained_path, 0.2)]},
                                         broken[1]: {0.01: [(high_val_pretrained_path, 0.5),
                                                            (low_val_pretrained_path, 0.2)]}}

        run_complete.run(source, target, broken, domain_tradeoff, dropout,
                         record_embeddings, pretraining_reps, best_only, gpu)

        mock_pretraining.assert_called_with(source, target, broken, domain_tradeoff, dropout,
                                            record_embeddings, pretraining_reps, gpu)
        dann_calls = [mock.call(source, target, broken[0], 1.0, record_embeddings, 1343278, gpu,
                                high_val_pretrained_path),
                      mock.call(source, target, broken[0], 1.0, record_embeddings, 9525887, gpu,
                                low_val_pretrained_path),
                      mock.call(source, target, broken[1], 1.0, record_embeddings, 1343278, gpu,
                                high_val_pretrained_path),
                      mock.call(source, target, broken[1], 1.0, record_embeddings, 9525887, gpu,
                                low_val_pretrained_path)
                      ]
        mock_dann.assert_has_calls(dann_calls)

    @mock.patch('run_complete.run_daan')
    @mock.patch('run_complete.run_pretraining')
    def test_run_with_all_pretrained(self, mock_pretraining, mock_dann):
        source = 3
        target = 1
        broken = [0.6, 0.4]
        domain_tradeoff = [0.1]
        dropout = 0.1
        record_embeddings = False
        pretraining_reps = 2
        best_only = True
        gpu = 1
        high_val_pretrained_path = 'high_val'
        low_val_pretrained_path = 'low_val'
        mock_pretraining.return_value = {broken[0]: {0.01: [(high_val_pretrained_path, 0.5),
                                                            (low_val_pretrained_path, 0.2)]},
                                         broken[1]: {0.01: [(high_val_pretrained_path, 0.5),
                                                            (low_val_pretrained_path, 0.2)]}}

        run_complete.run(source, target, broken, domain_tradeoff, dropout,
                         record_embeddings, pretraining_reps, best_only, gpu)

        mock_pretraining.assert_called_with(source, target, broken, domain_tradeoff, dropout,
                                            record_embeddings, pretraining_reps, gpu)
        dann_calls = [mock.call(source, target, broken[0], 1.0, record_embeddings, 1343278, gpu,
                                low_val_pretrained_path),
                      mock.call(source, target, broken[0], 1.0, record_embeddings, 9525887, gpu,
                                low_val_pretrained_path),
                      mock.call(source, target, broken[1], 1.0, record_embeddings, 1343278, gpu,
                                low_val_pretrained_path),
                      mock.call(source, target, broken[1], 1.0, record_embeddings, 9525887, gpu,
                                low_val_pretrained_path)
                      ]
        mock_dann.assert_has_calls(dann_calls)
