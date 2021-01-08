import json
import os

from lightning import loggers
from hyperopt.hyperopt_transfer import optimize_transfer
from hyperopt.hyperopt_pretraining import optimize_pretraining


def tune_complete(source, target, percent_broken, num_trials):
    transfer_config = optimize_transfer(source, target, percent_broken, num_trials)
    _save_transfer_config(transfer_config, source, target, percent_broken)
    pre_config = optimize_pretraining(
        source, target, percent_broken, transfer_config, num_trials
    )
    _save_pre_config(pre_config, source, target, percent_broken)


def _save_transfer_config(transfer_config, source, target, percent_broken):
    experiment_name = loggers.transfer_hyperopt_name(source, target, percent_broken)
    file_name = f"transfer_{experiment_name}.json"
    _save_config(transfer_config, file_name)


def _save_pre_config(pre_config, source, target, percent_broken):
    experiment_name = loggers.pretraining_hyperopt_name(source, target, percent_broken)
    file_name = f"transfer_{experiment_name}.json"
    _save_config(pre_config, file_name)


def _save_config(config, file_name):
    log_dir = os.path.join(_get_hyperopt_logdir(), "best_configs")
    os.makedirs(log_dir, exist_ok=True)
    save_path = os.path.join(log_dir, file_name)
    with open(save_path, mode="wt") as f:
        json.dump(config, f, indent=4)


def _get_hyperopt_logdir():
    script_path = os.path.dirname(__file__)
    log_dir = os.path.normpath(os.path.join(script_path, "..", "hyperopt"))

    return log_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization for complete process"
    )
    parser.add_argument("--source", type=int, required=True, help="source FD number")
    parser.add_argument("--target", type=int, required=True, help="target FD number")
    parser.add_argument(
        "--percent_broken", type=float, required=True, help="degradation in [0, 1]"
    )
    parser.add_argument(
        "--num_trials", type=int, required=True, help="number of hyperopt trials"
    )
    opt = parser.parse_args()

    tune_complete(opt.source, opt.target, opt.percent_broken, opt.num_trials)
