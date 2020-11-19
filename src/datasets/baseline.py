from typing import Optional, Union, List

import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

from datasets.adaption import _unify_source_and_target_length
from datasets.cmapss import CMAPSSDataModule


class BaselineDataModule(pl.LightningDataModule):
    def __init__(self,
                 fd_source,
                 batch_size,
                 max_rul=125,
                 window_size=30,
                 feature_select=None):
        super().__init__()

        self.fd_source = fd_source
        self.batch_size = batch_size
        self.window_size = window_size
        self.max_rul = max_rul
        self.feature_select = feature_select

        self.hparams = {'fd_source': self.fd_source,
                        'batch_size': self.batch_size,
                        'window_size': self.window_size,
                        'max_rul': self.max_rul}

        self.cmapss = {}
        for fd in range(1, 5):
            self.cmapss[fd] = CMAPSSDataModule(fd, batch_size, max_rul, window_size,
                                               None, None, feature_select)

    def prepare_data(self, *args, **kwargs):
        for cmapss_fd in self.cmapss.values():
            cmapss_fd.prepare_data(*args, **kwargs)

    def setup(self, stage: Optional[str] = None):
        for cmapss_fd in self.cmapss.values():
            cmapss_fd.setup(stage)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self._to_dataset(self.fd_source, split='dev'),
                          batch_size=self.batch_size,
                          shuffle=True,
                          pin_memory=True)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self._to_dataset(self.fd_source, split='val'),
                          batch_size=self.batch_size,
                          shuffle=False,
                          pin_memory=True)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        test_dataloaders = []
        for fd_target in range(1, 5):
            target_dl = DataLoader(self._to_dataset(self.fd_source, fd_target, 'test'),
                                   batch_size=self.batch_size,
                                   shuffle=False,
                                   pin_memory=True)
            test_dataloaders.append(target_dl)

        return test_dataloaders

    def _to_dataset(self, fd_source, fd_target=None, split=None):
        if split == 'dev' or split == 'val':
            data = self.cmapss[fd_source].data[split]
        elif split == 'test':
            source, source_labels = self.cmapss[fd_source].data[split]
            target, target_labels = self.cmapss[fd_target].data[split]
            data = _unify_source_and_target_length(source, source_labels, target, target_labels)
        else:
            raise ValueError(f'Invalid split {split}')

        dataset = TensorDataset(*data)

        return dataset
