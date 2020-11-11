import torch


class DataHparamsMixin:
    def add_data_hparams(self, data):
        self.hparams.update(data.hparams)


class LoadEncoderMixin:
    def load_encoder(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        encoder_state = {n.replace('encoder.', ''): weight
                         for n, weight in checkpoint['state_dict'].items()
                         if n.startswith('encoder')}
        self.encoder.load_state_dict(encoder_state)
        self.hparams['pretrained_checkpoint'] = checkpoint_path
