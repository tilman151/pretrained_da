"""Exports all RUL experiments to a data frame."""

import argparse
import re

import mlflow
import numpy as np
import pandas as pd


def export_all(mlflow_uri):
    """Export all transfer and baseline experiments to CSV."""
    client = mlflow.tracking.MlflowClient(mlflow_uri)

    print("Export transfer experiments...")
    df = _export_matching_experiments(client, ".+2.+$", _runs_of_transfer)
    if df is not None:
        df.to_csv("transfer.csv")

    print("Export baseline experiments...")
    df = _export_matching_experiments(
        client, "cmapss_.{3,5}_baseline$", _runs_of_baseline
    )
    if df is not None:
        df.to_csv("baseline.csv")


def _export_matching_experiments(client, exp_name_regex, func):
    regex = re.compile(exp_name_regex)
    experiments = client.list_experiments()
    filtered_experiments = [e for e in experiments if regex.match(e.name) is not None]
    print("Found %d experiments..." % len(filtered_experiments))
    df = []
    for e in filtered_experiments:
        print('Evaluate experiment "%s" with id %s' % (e.name, e.experiment_id))
        df.append(func(client, e))
    if df:
        df = pd.concat(df)
    else:
        df = None

    return df


def _runs_of_baseline(client, e):
    replications = _get_replications(client, e)
    df = pd.DataFrame(
        np.zeros((len(replications), 4)),
        columns=["mse_1", "mse_2", "mse_3", "mse_4"],
        index=[e.name] * len(replications),
    )
    for i, run in enumerate(replications):
        best_rmse = [
            _get_test_value(client, f"regression_loss_fd{i}", run) for i in range(1, 5)
        ]
        df.iloc[i] = best_rmse
    print("Return %d runs..." % len(df))

    return df


def _runs_of_transfer(client, experiment):
    """Retrieve and evaluate RUL experiment."""
    replications = _get_replications(client, experiment)
    replications = sorted(replications, key=lambda r: -r.info.start_time)[:50]
    df = pd.DataFrame(
        np.zeros((len(replications), 4)),
        columns=["percent_broken", "percent_fail_runs", "mse", "val_mse"],
        index=[f"cmapss_{experiment.name}_dann"] * len(replications),
    )
    for i, run in enumerate(replications):
        test_rmse = _get_test_value(client, "regression_loss", run)
        val_rmse = _get_val_value(client, "regression_loss", run)
        if (
            "percent_broken" in run.data.params
            and not run.data.params["percent_broken"] == "None"
        ):
            percent_broken = run.data.params["percent_broken"]
        else:
            percent_broken = 0.0
        if (
            "percent_fail_runs" in run.data.params
            and not run.data.params["percent_fail_runs"] == "None"
        ):
            percent_fail_runs = run.data.params["percent_fail_runs"]
        else:
            percent_fail_runs = 0.0
        df.iloc[i] = [percent_broken, percent_fail_runs, test_rmse, val_rmse]

    df = df.sort_values(["percent_broken", "percent_fail_runs"])
    print("Return %d runs..." % len(df))

    return df


def _get_replications(client, experiment):
    runs = client.search_runs(experiment_ids=experiment.experiment_id)
    print("Found %d top-level runs..." % len(runs))
    print("Found %d replications each..." % len(runs))

    return runs


def _get_test_value(client, metric, run):
    """Return test value of selected metric."""
    best_value = client.get_run(run.info.run_id).data.metrics[f"test/{metric}"]

    return best_value


def _get_val_value(client, metric, run):
    """Return test value of selected metric."""
    best_value = client.get_metric_history(run.info.run_id, f"val/{metric}")
    best_value = best_value[-1].value

    return best_value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exports all experiments to CSV")
    parser.add_argument("mlflow_uri", help="URI for MLFlow Server.")
    opt = parser.parse_args()

    export_all(opt.mlflow_uri)
