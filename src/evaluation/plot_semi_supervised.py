import pandas as pd
import plotnine
from plotnine import ggplot, aes


def plot(file_path):
    df = pd.read_csv(file_path, index_col=0)
    percent_broken = [0.6, 0.8]

    plotnine.options.figure_size = (15, 8)
    for broken in percent_broken:
        plot_df = df[(df["percent_broken"] == broken) | (df["percent_broken"] == 0.0)]
        gg = _box_plot(plot_df, "val")
        gg.save(file_path.replace(".csv", f"_val@{broken:.2f}.pdf"))
        gg = _box_plot(plot_df, "test")
        gg.save(file_path.replace(".csv", f"_test@{broken:.2f}.pdf"))


def _box_plot(df, column):
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
        + plotnine.scale_color_manual(["#14639e", "#DB5F57"])
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
