import os

import numpy as np
import pandas as pd
import plotnine
import statsmodels.formula.api as smf
from tabulate import tabulate

task_dict = {
    "four2three": "4→3",
    "four2two": "4→2",
    "one2three": "1→3",
    "one2two": "1→2",
    "three2four": "3→4",
    "three2one": "3→1",
    "two2four": "2→4",
    "two2one": "2→1",
    "one2four": "1→4",
    "four2one": "4→1",
    "two2three": "2→3",
    "three2two": "3→2",
}


def load_data(result_dir, filter_methods=None, filter_outlier=True, split="test"):
    transfer_df = _load_transfer(result_dir, filter_methods, filter_outlier, split)
    baseline_mse, baseline_rul = _load_baseline(result_dir)

    transfer_df["task"] = transfer_df["task"].map(task_dict)
    transfer_df = transfer_df.dropna(axis=0)
    baseline_mse["task"] = baseline_mse["task"].map(task_dict)
    if baseline_rul is not None:
        baseline_rul["task"] = baseline_rul["task"].map(task_dict)

    return transfer_df, baseline_rul, baseline_mse


def _load_transfer(result_dir, filter_methods, filter_outlier, split):
    transfer_path = os.path.join(result_dir, "transfer.csv")
    df = pd.read_csv(transfer_path, index_col=0)
    mse_name = "mse" if split == "test" else "val_mse"

    # Transform to percentages
    df["percent_broken"] *= 100
    df["percent_fail_runs"] *= 100

    # Add log of RUL score if score is recorded
    if "rul_score" in df.columns:
        df["log_score"] = df["rul_score"].apply(np.log)

    # Extract transfer task and transfer method from index (cmapss_<task>_<method>)
    task = df.index.map(lambda x: x.split("_")[1])
    method = df.index.map(lambda x: x.split("_")[2])
    df["task"] = task.values
    df["method"] = method.values

    if filter_methods is not None:
        df = df[df["method"].isin(filter_methods)]

    if filter_outlier:
        grouped = df.groupby(["task", "percent_broken"])
        quantiles = grouped.agg(
            q1=(mse_name, lambda x: x.quantile(0.25)),
            q3=(mse_name, lambda x: x.quantile(0.75)),
        )
        iqr = quantiles["q3"] - quantiles["q1"]
        thresholds = quantiles["q3"] + 1.5 * iqr
        sub_dfs = []
        for group in grouped.groups:
            sub_df = grouped.get_group(group)
            sub_dfs.append(sub_df[sub_df[mse_name] < thresholds[group]])
        df = pd.concat(sub_dfs)
        df = df[df[mse_name] < 160]

    if mse_name == "mse":
        df = df.drop("val_mse", axis=1)
    else:
        df["mse"] = df[mse_name]
        df = df.drop(mse_name)

    return df


def _load_baseline(result_dir):
    baseline_path = os.path.join(result_dir, "baseline.csv")
    baseline = pd.read_csv(baseline_path, index_col=0)

    baseline = pd.DataFrame(baseline.stack())
    baseline.columns = ["value"]
    baseline["dataset"] = baseline.index.codes[0]
    baseline["measure"] = baseline.index.codes[1]
    baseline["type"] = baseline["measure"] > 3
    baseline["measure"] = baseline["measure"] % 4
    baseline.loc["cmapss_one_baseline", "dataset"] = 0
    baseline.loc["cmapss_two_baseline", "dataset"] = 1
    baseline.loc["cmapss_three_baseline", "dataset"] = 2
    baseline.loc["cmapss_four_baseline", "dataset"] = 3

    # Remove non-transfer (e.g. one2one)
    baseline = baseline[baseline["dataset"] != baseline["measure"]]

    codes2dataset = {0: "one", 1: "two", 2: "three", 3: "four"}
    task = [
        f'{codes2dataset[x["dataset"]]}2{codes2dataset[x["measure"]]}'
        for _, x in baseline.iterrows()
    ]
    baseline = baseline.assign(task=task)

    if baseline["type"].any():
        baseline_mse = baseline[baseline["type"] == 1]
        baseline_rul = baseline[baseline["type"] == 0]
        baseline_rul["log_value"] = baseline_rul["value"].apply(np.log)
    else:
        baseline_mse = baseline[baseline["type"] == 0]
        baseline_rul = None

    return baseline_mse, baseline_rul


def mixed_linear_plots(df, x_axis, x_label):
    plotnine.options.figure_size = (8, 10)

    if "log_score" in df.columns:
        md = smf.mixedlm("log_score ~ percent_broken", df, groups=df.index.values)
        mdf_rul = md.fit()

        print("#" * 18 + "Log RUL" + "#" * 18)
        print(mdf_rul.summary())

    md = smf.mixedlm("mse ~ percent_broken", df, groups=df.index.values)
    mdf_mse = md.fit()

    print("#" * 18 + "RMSE" + "#" * 18)
    print(mdf_mse.summary())

    df["percent_broken"] = df["percent_broken"].round().astype(np.int)
    df["percent_fail_runs"] = df["percent_fail_runs"].round().astype(np.int)

    if "log_score" in df.columns:
        gg = (
            plotnine.ggplot(df, plotnine.aes(x=x_axis, y="log_score", color="method"))
            + plotnine.geom_jitter(width=2.5, show_legend=False)
            + plotnine.geom_abline(
                plotnine.aes(
                    intercept=mdf_rul.params["Intercept"], slope=mdf_rul.params[x_axis]
                )
            )
            + plotnine.stat_smooth(method="gls", show_legend=False)
            + plotnine.xlab(x_label)
            + plotnine.ylab("Logarithmic RUL-Score")
            + plotnine.scale_color_discrete(name="Method")
            + plotnine.theme_classic(base_size=20)
        )
        gg.save("%s_log_rul_by_method.pdf" % x_axis)

        gg = (
            plotnine.ggplot(df, plotnine.aes(x=x_axis, y="log_score", color="task"))
            + plotnine.geom_jitter(width=2.5, show_legend=False)
            + plotnine.geom_abline(
                plotnine.aes(
                    intercept=mdf_rul.params["Intercept"], slope=mdf_rul.params[x_axis]
                )
            )
            + plotnine.stat_smooth(method="gls", show_legend=False)
            + plotnine.xlab(x_label)
            + plotnine.ylab("Logarithmic RUL-Score")
            + plotnine.scale_color_discrete(name="Task")
            + plotnine.theme_classic(base_size=20)
        )
        gg.save("%s_log_rul_by_task.pdf" % x_axis)

    gg = (
        plotnine.ggplot(df, plotnine.aes(x=x_axis, y="mse", color="method"))
        + plotnine.geom_jitter(width=2.5)
        + plotnine.geom_abline(
            plotnine.aes(
                intercept=mdf_mse.params["Intercept"], slope=mdf_mse.params[x_axis]
            )
        )
        + plotnine.stat_smooth(method="gls")
        + plotnine.ylab("RMSE")
        + plotnine.xlab(x_label)
        + plotnine.scale_color_discrete(name="Method")
        + plotnine.theme_classic(base_size=20)
    )
    gg.save("%s_mse_by_method.pdf" % x_axis)

    gg = (
        plotnine.ggplot(df, plotnine.aes(x=x_axis, y="mse", color="task"))
        + plotnine.geom_jitter(width=2.5)
        + plotnine.geom_abline(
            plotnine.aes(
                intercept=mdf_mse.params["Intercept"], slope=mdf_mse.params[x_axis]
            )
        )
        + plotnine.stat_smooth(method="gls")
        + plotnine.ylab("RMSE")
        + plotnine.scale_color_discrete(name="Task")
        + plotnine.theme_classic(base_size=20)
    )
    gg.save("%s_mse_by_task.pdf" % x_axis)


def mixed_linear_factors_plot(df, x_axis, factor):
    plotnine.options.figure_size = (10, 10)
    factor_steps = df[factor].unique()
    reg_lines = pd.DataFrame(
        {
            factor: factor_steps,
            "intercept": np.zeros_like(factor_steps),
            "slope": np.zeros_like(factor_steps),
        }
    )
    for i, step in enumerate(factor_steps):
        factored_df = df[df[factor] == step]
        md = smf.mixedlm(
            "mse ~ %s" % x_axis, factored_df, groups=factored_df.index.values
        )
        mdf = md.fit()
        reg_lines.iloc[i] = [step, mdf.params["Intercept"], mdf.params[x_axis]]

    df["percent_broken"] = df["percent_broken"].round().astype(np.int)
    df["percent_fail_runs"] = df["percent_fail_runs"].round().astype(np.int)
    reg_lines[factor] = reg_lines[factor].round().astype(np.int)
    gg = (
        plotnine.ggplot(df, plotnine.aes(x=x_axis, y="mse", color="method"))
        + plotnine.geom_jitter(width=2.5, show_legend=False)
        + plotnine.scale_color_manual(["#DB5F57"] * 4)
        + plotnine.facet_wrap(factor)
        + plotnine.geom_abline(
            plotnine.aes(intercept="intercept", slope="slope"), data=reg_lines
        )
        + plotnine.theme_classic(base_size=20)
    )
    gg.save("%s_vs_%s_rmse.pdf" % (x_axis, factor))


def method_plot(df, baseline_rul, baseline_mse, method):
    plotnine.options.figure_size = (15, 8)

    method_df = df[df["method"] == method]

    method_df["percent_broken"] = method_df["percent_broken"].round().astype(np.int)
    method_df["percent_fail_runs"] = method_df["percent_fail_runs"].round().astype(np.int)

    if baseline_rul is not None:
        plotnine.ylim = (2, 10)
        gg = (
            plotnine.ggplot(
                method_df,
                plotnine.aes(x="percent_broken", y="log_score", color="method"),
            )
            + plotnine.facet_wrap("task", 3, 4)
            + plotnine.stat_boxplot(
                plotnine.aes(y="log_value", x=60),
                data=baseline_rul,
                width=80,
                color="#14639e",
                show_legend=False,
            )
            + plotnine.geom_jitter(width=2.5, show_legend=False)
            + plotnine.stat_smooth(method="gls", show_legend=False)
            + plotnine.xlab("Grade of Degradation in %")
            + plotnine.ylab("Logarithmic RUL-Score")
            + plotnine.theme_classic(base_size=20)
        )
        gg.save("%s_log_rul.pdf" % method)

    plotnine.ylim = (90, 10)
    gg = (
        plotnine.ggplot(
            method_df, plotnine.aes(x="percent_broken", y="mse", color="method")
        )
        + plotnine.facet_wrap("task", 3, 4)
        + plotnine.stat_boxplot(
            plotnine.aes(y="value", x=60),
            data=baseline_mse,
            width=80,
            color="#14639e",
            show_legend=False,
        )
        + plotnine.geom_jitter(width=2.5, show_legend=False)
        + plotnine.stat_smooth(method="gls", show_legend=False)
        + plotnine.xlab("Grade of Degradation in %")
        + plotnine.ylab("RMSE")
        + plotnine.theme_classic(base_size=20)
    )
    gg.save("%s_rmse.pdf" % method)


def method_tables(df: pd.DataFrame, baseline_rul: pd.DataFrame, baseline_mse, method):
    filtered_df = df[df["method"] == method]
    filtered_df.loc[:, "percent_broken"] = (
        filtered_df["percent_broken"].round().astype(np.int)
    )
    filtered_df.loc[:, "percent_fail_runs"] = (
        filtered_df["percent_fail_runs"].round().astype(str)
    )

    filtered_df = filtered_df.drop(columns=["percent_fail_runs"])
    means = filtered_df.groupby(["task", "percent_broken"]).mean()
    stds = filtered_df.groupby(["task", "percent_broken"]).std()

    mse = _build_table(baseline_mse, means, stds, "mse")
    print("RMSE:")
    print(tabulate(mse, headers="keys", tablefmt="latex_raw"))

    if baseline_rul is not None:
        rul = _build_table(baseline_rul, means, stds, "rul_score")
        print()
        print("RUL:")
        print(tabulate(rul, headers="keys", tablefmt="latex_raw"))


def _build_table(baseline_mse, means, stds, metric_name):
    metric_table = means[metric_name].combine(
        stds[metric_name], func=lambda mean, std: "$%.2f \pm %.2f$" % (mean, std)
    )
    metric_table = metric_table.unstack(level=-1).sort_index(level=1, axis=1)
    baseline_mse = baseline_mse.drop(columns=["dataset", "measure", "type"])
    baseline_mse_means = baseline_mse.groupby("task").mean()
    baseline_mse_stds = baseline_mse.groupby("task").std()
    baseline_mse = baseline_mse_means["value"].combine(
        baseline_mse_stds["value"],
        func=lambda mean, std: "$%.2f \pm %.2f$" % (mean, std),
    )
    baseline_mse = pd.DataFrame(baseline_mse)
    baseline_mse.columns = ["source only"]
    metric_table = pd.concat([baseline_mse, metric_table], axis=1)

    return metric_table


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot results of transfer against baselines."
    )
    parser.add_argument(
        "result_dir", help="path to folder with baseline.csv and transfer.csv"
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
        help="plot validation or test results",
    )
    parser.add_argument(
        "--filter_outlier",
        action="store_true",
        help="whether to filter out outlier transfer runs",
    )
    opt = parser.parse_args()

    transfer, base_rul, base_rmse = load_data(
        opt.result_dir,
        filter_methods=["dann", "dannPre"],
        filter_outlier=opt.filter_outlier,
        split=opt.split,
    )
    mixed_linear_plots(transfer, "percent_broken", "Grade of Degradation in %")
    # mixed_linear_plots(transfer, 'percent_fail_runs', 'Number of Systems in %')
    # mixed_linear_factors_plot(transfer, 'percent_fail_runs', 'percent_broken')
    # mixed_linear_factors_plot(transfer, 'percent_broken', 'percent_fail_runs')
    for m in ["dann", "dannPre"]:
        method_plot(transfer, base_rul, base_rmse, m)
    for m in ["dann", "dannPre"]:
        print("")
        print("#" * 18 + m + "#" * 18)
        print("")
        method_tables(transfer, base_rul, base_rmse, m)
