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