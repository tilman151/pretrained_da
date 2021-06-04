import pandas as pd
import plotnine
from plotnine import ggplot, aes
from matplotlib import cm, colors


def plot(file_path):
    df = pd.read_csv(file_path, index_col=0)
    df["pretrained"] = pd.Categorical(
        df["pretrained"], ["no pretraining", "metric", "autoencoder", "rbm"], ordered=True
    )
    percent_broken = [0.4, 0.6, 0.8]

    plotnine.options.figure_size = (15, 8)
    for broken in percent_broken:
        plot_df = df[(df["percent_broken"] == broken) | (df["percent_broken"] == 0.0)]
        baseline_metric = plot_df[
            (plot_df["pretrained"] == "metric")
            | (plot_df["pretrained"] == "no pretraining")
        ]
        _plot_val_test(baseline_metric, broken, file_path)
        comparison = plot_df[plot_df["source"] == 4]
        _plot_val_test(
            comparison,
            broken,
            file_path.replace("semi_supervised.csv", "ssl_comparison.csv"),
        )


def _plot_val_test(plot_df, broken, file_path):
    gg = _box_plot(plot_df, "val")
    gg.save(file_path.replace(".csv", f"_val@{broken:.2f}.pdf"))
    gg = _box_plot(plot_df, "test")
    gg.save(file_path.replace(".csv", f"_test@{broken:.2f}.pdf"))


def _box_plot(df, column):
    df = df.sort_values("pretrained")
    gg = (
        ggplot(df, aes(x="num_labeled", y=column))
        + plotnine.stat_boxplot(
            aes(
                y=column,
                x="factor(num_labeled)",
                color="pretrained",
            )
        )
        + plotnine.facet_wrap("source", nrow=2, ncol=2, scales="free_x")
        + plotnine.scale_color_manual(
            [colors.rgb2hex(c) for c in cm.get_cmap("tab10").colors]
        )
        + plotnine.xlab("Number of Labeled Runs")
        + plotnine.ylab("RMSE")
        + plotnine.theme_classic(base_size=20)
        + plotnine.theme(subplots_adjust={"hspace": 0.25})
    )

    return gg


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot results of semi-supervised experiments"
    )
    parser.add_argument("file_path", help="path to result CSV file")
    opt = parser.parse_args()

    plot(opt.file_path)
