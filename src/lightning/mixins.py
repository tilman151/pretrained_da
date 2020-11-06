class DataHparamsMixin:
    def add_data_hparams(self, data):
        self.hparams.update(data.hparams)
