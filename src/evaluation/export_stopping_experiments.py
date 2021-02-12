import mlflow
import numpy as np
import pandas as pd

from evaluation.export_results import export_matching_experiments


def export_stopping_experiments(mlflow_uri):
    client = mlflow.tracking.MlflowClient(mlflow_uri)
    df = export_matching_experiments(client, ".+2.+$", _stopping_experiments)
    if df is not None:
        df.to_csv("stopping.csv")


def _stopping_experiments(client, e):
    runs = client.search_runs(experiment_ids=e.experiment_id)
    df = pd.DataFrame(
        np.zeros((len(runs), 11)),
        columns=[
            "index",
            "source",
            "target",
            "percent_broken",
            "target/mse",
            "score/mse",
            "source/mse",
            "target/rul",
            "score/rul",
            "source/rul",
            "version",
        ],
    )

    for i, r in enumerate(runs):
        row = [
            e.name,
            r.data.params["fd_source"],
            r.data.params["fd_target"],
            r.data.params["percent_broken"],
            r.data.metrics["test/regression_loss/regression_loss"],
            r.data.metrics["test/score/regression_loss"],
            r.data.metrics["test/source_regression_loss/regression_loss"],
            r.data.metrics["test/regression_loss/rul_score"],
            r.data.metrics["test/score/rul_score"],
            r.data.metrics["test/source_regression_loss/rul_score"],
            r.data.tags["version"],
        ]
        df.iloc[i] = row

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export experiments with different stopping metrics."
    )
    parser.add_argument("mlflow_uri", help="URI for mlflow")
    opt = parser.parse_args()

    export_stopping_experiments(opt.mlflow_uri)
