import unittest

import torch

import lightning


class TestAdaptiveAE(unittest.TestCase):
    def setUp(self):
        self.trade_off = 0.5
        self.net = lightning.AdaptiveAE(in_channels=14,
                                        seq_len=30,
                                        num_layers=4,
                                        kernel_size=3,
                                        base_filters=16,
                                        latent_dim=64,
                                        recon_trade_off=self.trade_off,
                                        domain_trade_off=self.trade_off,
                                        domain_disc_dim=32,
                                        num_disc_layers=2,
                                        source_rul_cap=50,
                                        optim_type='adam',
                                        lr=0.01)

    def test_encoder(self):
        inputs = torch.randn(16, 14, 30)
        outputs = self.net.encoder(inputs)
        self.assertEqual(torch.Size((16, 64)), outputs.shape)

    def test_decoder(self):
        inputs = torch.randn(16, 64)
        outputs = self.net.decoder(inputs)
        self.assertEqual(torch.Size((16, 14, 30)), outputs.shape)

    def test_classifier(self):
        inputs = torch.randn(16, 64)
        outputs = self.net.classifier(inputs)
        self.assertEqual(torch.Size((16, 1)), outputs.shape)

    def test_domain_disc(self):
        inputs = torch.randn(16, 64)
        outputs = self.net.domain_disc(inputs)
        self.assertEqual(torch.Size((16, 1)), outputs.shape)

    def test_trade_off(self):
        inputs = (torch.randn(16, 14, 30), torch.randn(16), torch.randn(16, 14, 30))
        loss = self.net.training_step(inputs, 0)
        combined_loss = (loss['train/regression_loss'] +
                         self.trade_off * loss['train/recon_loss'] +
                         self.trade_off * loss['train/domain_loss'])
        self.assertEqual(loss['train/loss'], combined_loss)

    def test_batch_independence(self):
        inputs = torch.randn(16, 14, 30)
        inputs.requires_grad = True

        # Compute forward pass in eval mode to deactivate batch norm
        self.net.eval()
        outputs = self.net(inputs)
        self.net.train()

        for n, output in enumerate(outputs):
            with self.subTest(n_output=n):
                # Mask loss for certain samples in batch
                batch_size = output.shape[0]
                mask_idx = torch.randint(0, batch_size, ())
                mask = torch.ones_like(output)
                mask[mask_idx] = 0
                output = output * mask

                # Compute backward pass
                loss = output.mean()
                loss.backward(retain_graph=True)

                # Check if gradient exists and is zero for masked samples
                for i, grad in enumerate(inputs.grad[:batch_size]):
                    if i == mask_idx:
                        self.assertTrue(torch.all(grad == 0).item())
                    else:
                        self.assertTrue(not torch.all(grad == 0))
                inputs.grad = None

    def test_all_parameters_updated(self):
        optim = torch.optim.SGD(self.net.parameters(), lr=0.1)

        outputs = self.net(torch.randn(16, 14, 30))
        loss = sum(o.mean() for o in outputs)
        loss.backward()
        optim.step()

        for param_name, param in self.net.named_parameters():
            if param.requires_grad:
                with self.subTest(name=param_name):
                    self.assertIsNotNone(param.grad)
                    self.assertNotEqual(0., torch.sum(param.grad ** 2))

    def test_eval_metrics(self):
        criterion = torch.nn.MSELoss()
        target = torch.zeros(16, 14, 30)
        source = torch.zeros(16, 14, 30)
        target_labels = torch.ones(16)
        domain_labels = torch.cat([torch.zeros_like(target_labels),
                                   torch.ones_like(target_labels)])

        expected_prediction = self.net.classifier(self.net.encoder(target))
        expected_loss = criterion(expected_prediction.squeeze(), target_labels)
        _, _, actual_loss, _ = self.net._calc_loss(target, target_labels, source, domain_labels)

        self.assertEqual(expected_loss, actual_loss)

    def test_get_rul_mask(self):
        labels = torch.arange(0, 125)
        features = torch.randn(250, 20)
        capped_rul_mask = self.net._get_rul_mask(labels, cap=True)
        uncapped_rul_mask = self.net._get_rul_mask(labels, cap=False)

        self.assertEqual(250, capped_rul_mask.shape[0])  # mask has double batch size
        self.assertFalse(capped_rul_mask[:51].all().item())  # mask is False for <= 50 RUL
        self.assertTrue(capped_rul_mask[51:125].all().item())  # mask is True for > 50
        self.assertFalse(capped_rul_mask[125:176].all().item())  # mask is repeated two times
        self.assertTrue(capped_rul_mask[176:].all().item())
        self.assertEqual(148, features[capped_rul_mask].shape[0])  # mask indexes correct number of samples

        self.assertEqual(250, uncapped_rul_mask.shape[0])  # mask has double batch size
        self.assertTrue(uncapped_rul_mask.all().item())  # all samples are True
        self.assertEqual(250, features[uncapped_rul_mask].shape[0])  # all samples are indexed

