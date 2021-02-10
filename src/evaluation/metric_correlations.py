import mlflow
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


def calc_correlation_between_metrics(run_id, metrics, mlflow_uri):
    client = mlflow.tracking.MlflowClient(mlflow_uri)
    metric_histories = {}
    for metric in metrics:
        metric_name = f"val/{metric}"
        history = client.get_metric_history(run_id, metric_name)
        metric_histories[metric] = np.array([step.value for step in history])

    for first in metrics:
        for second in metrics:
            print(
                f"{first} vs. {second}: ",
                scipy.stats.spearmanr(metric_histories[first], metric_histories[second]),
            )
            plt.scatter(metric_histories[first], metric_histories[second])
            plt.savefig(f"{first}_vs_{second}.png")
            plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate the correlation between the validation metrics of a run."
    )
    parser.add_argument("run_id", help="id of run to analyze")
    parser.add_argument(
        "--metrics",
        default=["regression_loss", "score", "source_regression_loss"],
        help="list of metrics to compare",
    )
    parser.add_argument(
        "--mlflow_uri", default="http://localhost:5000", help="URI for mlflow"
    )
    opt = parser.parse_args()

    calc_correlation_between_metrics(opt.run_id, opt.metrics, opt.mlflow_uri)
