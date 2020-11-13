import os

import pytorch_lightning.loggers as loggers

ExperimentNaming = {1: 'one',
                    2: 'two',
                    3: 'three',
                    4: 'four'}


def baseline_experiment_name(source):
    assert source in ExperimentNaming, f'Unknown FD number {source}.'
    return f'cmapss_{ExperimentNaming[source]}_baseline'


def transfer_experiment_name(source, target):
    assert source in ExperimentNaming, f'Unknown source FD number {source}.'
    assert target in ExperimentNaming, f'Unknown target FD number {target}.'
    return f'{ExperimentNaming[source]}2{ExperimentNaming[target]}'


def pretraining_experiment_name(dataset1, dataset2):
    assert dataset1 in ExperimentNaming, f'Unknown dataset1 FD number {dataset1}.'
    assert dataset2 in ExperimentNaming, f'Unknown dataset2 FD number {dataset2}.'
    if dataset1 > dataset2:
        dataset1, dataset2 = dataset2, dataset2  # swap to use smaller FD first for consistency
    return f'pretraining_{ExperimentNaming[dataset1]}&{ExperimentNaming[dataset2]}'


class MLTBLogger(loggers.LoggerCollection):
    """Combined MlFlow and Tensorboard logger that saves models as MlFlow artifacts."""

    def __init__(self, log_dir, experiment_name, tensorboard_struct=None):
        """
        This logger combines a MlFlow and Tensorboard logger.
        It creates a directory (mlruns/tensorboard) for each logger in log_dir.

        If a tensorboard_struct dict is provided, it is used to create additional
        sub-directories for tensorboard to get a better overview over the runs.

        The difference to a simple LoggerCollection is that the save dir points
        to the artifact path of the MlFlow run. This way the model is logged as
        a MlFlow artifact.

        :param log_dir: directory to put the mlruns and tensorboard directories
        :param experiment_name: name for the experiment
        :param tensorboard_struct: dictionary containing information to refine the tensorboard directory structure
        """
        tensorboard_path = os.path.join(log_dir, 'tensorboard', experiment_name)
        sub_dirs = self._dirs_from_dict(tensorboard_struct)
        self._tf_logger = loggers.TensorBoardLogger(tensorboard_path, name=sub_dirs)

        mlflow_path = 'file:' + os.path.normpath(os.path.join(log_dir, 'mlruns'))
        self._mlflow_logger = loggers.MLFlowLogger(experiment_name, tracking_uri=mlflow_path)

        super().__init__([self._tf_logger, self._mlflow_logger])

    @staticmethod
    def _dirs_from_dict(tensorboard_struct):
        if tensorboard_struct is not None:
            dirs = []
            for key, value in tensorboard_struct.items():
                if isinstance(value, float) and value >= 0.1:
                    dirs.append(f'{value:.1f}{key}')
                elif isinstance(value, str):
                    dirs.append(f'{key}:{value}')
                else:
                    dirs.append(f'{value}{key}')
            dirs = os.path.join(*dirs)
        else:
            dirs = ''

        return dirs

    @property
    def name(self) -> str:
        return self._mlflow_logger.name

    @property
    def version(self) -> str:
        return os.path.join(self._mlflow_logger.version, 'artifacts')

    @property
    def save_dir(self):
        return self._mlflow_logger.save_dir
