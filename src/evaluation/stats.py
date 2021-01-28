import pandas as pd
from scipy.stats import wilcoxon
from tabulate import tabulate

from evaluation.plot_results import load_data


def calculate_statistics(transfer_df: pd.DataFrame, baseline_df):
    transfer_df = _get_mean_performances(transfer_df)
    baseline_df = _get_mean_baseline(baseline_df)
    pre_wins = _compare_transfer(transfer_df)
    baseline_losses = _compare_transfer_baseline(transfer_df, baseline_df)
    print("Pretrained better than DANN:")
    print(tabulate(pre_wins, headers="keys", tablefmt="latex_raw"))
    print("Transfer better than Baseline:")
    print(tabulate(baseline_losses, headers="keys", tablefmt="latex_raw"))


def _compare_transfer(transfer_df):
    pre_wins = pd.DataFrame({"p": [0.0] * 5}, index=[20.0, 40.0, 60.0, 80.0, 100.0])
    for percent_broken in [20.0, 40.0, 60.0, 80.0, 100.0]:
        dann = transfer_df.loc[percent_broken].loc["dann"].set_index("task")
        dann_pre = transfer_df.loc[percent_broken].loc["dannPre"].set_index("task")
        difference = (dann - dann_pre).dropna().values.squeeze()
        stat, p = wilcoxon(difference, alternative="greater")
        pre_wins.loc[percent_broken] = p

    return pre_wins


def _compare_transfer_baseline(transfer_df, baseline_df):
    baseline_losses = pd.DataFrame(
        {"p_pre": [0.0] * 5, "p_dann": [0.0] * 5}, index=[20.0, 40.0, 60.0, 80.0, 100.0]
    )
    for percent_broken in [20.0, 40.0, 60.0, 80.0, 100.0]:
        dann = transfer_df.loc[percent_broken].loc["dann"].set_index("task")
        dann_pre = transfer_df.loc[percent_broken].loc["dannPre"].set_index("task")
        difference_dann = (baseline_df - dann).dropna().values.squeeze()
        stat, p_dann = wilcoxon(difference_dann, alternative="greater")
        difference_pre = (baseline_df - dann_pre).dropna().values.squeeze()
        stat, p_pre = wilcoxon(difference_pre, alternative="greater")
        baseline_losses.loc[percent_broken] = [p_pre, p_dann]

    return baseline_losses


def _get_mean_baseline(baseline_df):
    baseline_df = (
        baseline_df.reset_index()
        .groupby(by=["task", "type"])["value"]
        .mean()
        .reset_index()
        .set_index("task")
        .rename(columns={"value": "mse"})
        .drop("type", axis=1)
    )
    return baseline_df


def _get_mean_performances(transfer_df):
    transfer_df = (
        transfer_df.reset_index()
        .groupby(by=["method", "task", "percent_broken"])["mse"]
        .mean()
        .reset_index()
    )
    transfer_df = transfer_df.set_index(["percent_broken", "method"])

    return transfer_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculate statistics about results.")
    parser.add_argument(
        "result_dir", help="path to folder with baseline.csv and transfer.csv"
    )
    opt = parser.parse_args()

    transfer, base_rul, base_rmse = load_data(
        opt.result_dir,
        filter_methods=["dann", "dannPre"],
        filter_outlier=False,
        split="test",
    )
    calculate_statistics(transfer, base_rmse)
