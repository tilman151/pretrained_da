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
        best_rmse = [_get_test_value(f"regression_loss_fd{i}", run) for i in range(1, 5)]
        df.iloc[i] = best_rmse
    print("Return %d runs..." % len(df))

    return df


def _runs_of_transfer(client, experiment):
    """Retrieve and evaluate RUL experiment."""
    replications = _get_replications(client, experiment)
    df = pd.DataFrame(
        np.zeros((len(replications), 6)),
        columns=[
            "index",
            "percent_broken",
            "percent_fail_runs",
            "mse",
            "val_mse",
            "version",
        ],
    )

    for i, run in enumerate(replications):
        test_rmse = _get_test_value("regression_loss", run)
        val_rmse = _get_val_value(
            client, "regression_loss", "source_regression_loss", run
        )
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
        version = run.data.tags["version"] if "version" in run.data.tags else ""
        if "pretrained_checkpoint" in run.data.params:
            index = f"cmapss_{experiment.name}_dann_pre"
        else:
            index = f"cmapss_{experiment.name}_dann"
        df.iloc[i] = [
            index,
            percent_broken,
            percent_fail_runs,
            test_rmse,
            val_rmse,
            version,
        ]

    df = df[df["version"] != ""]
    df = df.groupby("version").filter(lambda group: group["version"].count() == 50)
    df = df.sort_values(["percent_broken", "percent_fail_runs"])
    print("Return %d runs..." % len(df))

    return df


def _get_replications(client, experiment):
    runs = client.search_runs(experiment_ids=experiment.experiment_id)
    print("Found %d top-level runs..." % len(runs))

    return runs


def _get_test_value(metric, run):
    """Return test value of selected metric."""
    test_metric = f"test/{metric}"
    if test_metric in run.data.metrics:
        best_value = run.data.metrics[test_metric]
    else:
        best_value = np.nan

    return best_value


def _get_val_value(client, metric, indicator_metric, run):
    """Return test value of selected metric."""
    val_metric = f"val/{metric}"
    val_indicator = f"val/{indicator_metric}"
    if val_metric in run.data.metrics and val_indicator in run.data.metrics:
        indicator_values = client.get_metric_history(run.info.run_id, val_indicator)
        best_indicator_idx = np.argmin([m.value for m in indicator_values])
        metric_value = client.get_metric_history(run.info.run_id, val_metric)
        best_value = metric_value[best_indicator_idx].value
    else:
        best_value = np.nan

    return best_value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exports all experiments to CSV")
    parser.add_argument("mlflow_uri", help="URI for MLFlow Server.")
    opt = parser.parse_args()

    export_all(opt.mlflow_uri)
