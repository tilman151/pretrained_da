import plotnine
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from tabulate import tabulate

from evaluation.plot_results import task_dict


def plot_stopping_experiments(result_path):
    df = pd.read_csv(result_path, index_col=0)
    df["index"] = df["index"].map(task_dict)
    df = df.melt(id_vars=["index", "source", "target", "percent_broken", "version"])

    df = df[df["variable"].str.endswith("mse")]
    plotnine.options.figure_size = (15, 8)
    gg = (
        plotnine.ggplot(df, plotnine.aes(x="percent_broken", y="value"))
        + plotnine.facet_wrap("index", 3, 4)
        + plotnine.stat_boxplot(
            plotnine.aes(y="value", x="factor(percent_broken)", color="variable"),
        )
        + plotnine.scale_x_discrete(
            labels=["0.2", "0.4", "0.6", "0.8", "1.0"], name="Grade of Degradation in %"
        )
        + plotnine.labs(color="Stopping Metric")
        + plotnine.ylab("RMSE")
        + plotnine.theme_classic(base_size=20)
    )
    gg.save("target_mse.pdf")


def test_significance(result_path):
    df = pd.read_csv(result_path, index_col=0)
    df = df.groupby(by=["percent_broken", "index"]).mean()

    significances = pd.DataFrame(
        np.zeros((5, 2)), columns=["stat", "p"], index=[0.2, 0.4, 0.6, 0.8, 1.0]
    )
    for idx in significances.index:
        score_perf = df.loc[idx, "score/mse"].values.squeeze()
        source_perf = df.loc[idx, "source/mse"].values.squeeze()
        difference = score_perf - source_perf
        stat, p = wilcoxon(difference, alternative="less")
        significances.loc[idx] = [stat, p]

    print(tabulate(significances, headers="keys", tablefmt="latex_raw"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot results of stopping experiments.")
    parser.add_argument("result_path", help="path to the result CSV")
    opt = parser.parse_args()

    plot_stopping_experiments(opt.result_path)
    test_significance(opt.result_path)
