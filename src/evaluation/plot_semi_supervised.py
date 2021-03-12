import pandas as pd
import plotnine
from plotnine import ggplot, aes


def plot(file_path):
    df = pd.read_csv(file_path, index_col=0)

    gg = (
        ggplot(df, aes(x="num_labeled", y="val"))
        + plotnine.stat_boxplot(
            aes(
                y="val",
                x="factor(num_labeled)",
                color="pretrained",
            )
        )
        + plotnine.scale_color_manual(["#14639e", "#DB5F57"])
        + plotnine.xlab("Number of Labeled Runs")
        + plotnine.ylab("RMSE")
        + plotnine.theme_classic(base_size=20)
    )
    gg.save(file_path.replace(".csv", "_val.pdf"))

    gg = (
        ggplot(df, aes(x="num_labeled", y="rmse_4"))
        + plotnine.stat_boxplot(
            aes(
                y="rmse_4",
                x="factor(num_labeled)",
                color="pretrained",
            )
        )
        + plotnine.scale_color_manual(["#14639e", "#DB5F57"])
        + plotnine.xlab("Number of Labeled Runs")
        + plotnine.ylab("RMSE")
        + plotnine.theme_classic(base_size=20)
    )
    gg.save(file_path.replace(".csv", "_test.pdf"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot results of semi-supervised experiments"
    )
    parser.add_argument("file_path", help="path to result CSV file")
    opt = parser.parse_args()

    plot(opt.file_path)
