import matplotlib.colors as mplcolors
import matplotlib.pyplot as plt
import torch
import torchmetrics
import umap
from torch import nn as nn


class EmbeddingViz(torchmetrics.Metric):
    def __init__(self, embedding_size, combined=True):
        super().__init__()

        self.embedding_size = embedding_size
        self.combined = combined

        self.add_state("embeddings", default=[], dist_reduce_fx="cat")
        self.add_state("labels", default=[], dist_reduce_fx="cat")
        self.add_state("ruls", default=[], dist_reduce_fx="cat")

        self.class_cm = plt.get_cmap("tab10")
        self.rul_cm = plt.get_cmap("viridis")

    def update(self, embeddings, labels, ruls):
        """
        Add embedding, labels and RUL to the metric.

        :param embeddings: hidden layer embedding of the data points
        :param labels: domain labels with 0 being source and 1 being target
        :param ruls: Remaining Useful Lifetime values of the data points
        """

        self.embeddings.append(embeddings)
        self.labels.append(labels)
        self.ruls.append(ruls)

    def compute(self):
        """Compute UMAP and plot points to 2d scatter plot."""
        logged_embeddings = torch.cat(self.embeddings).detach().cpu().numpy()
        logged_labels = torch.cat(self.labels).detach().cpu().int()
        logged_ruls = torch.cat(self.ruls).detach().cpu()
        viz_embeddings = umap.UMAP(random_state=42).fit_transform(logged_embeddings)

        if self.combined:
            fig, (ax_class, ax_rul) = plt.subplots(
                1, 2, sharex="all", sharey="all", figsize=(20, 10), dpi=300
            )
        else:
            fig_class = plt.figure(figsize=(10, 10), dpi=300)
            ax_class = fig_class.gca()
            fig_rul = plt.figure(figsize=(10, 10), dpi=300)
            ax_rul = fig_rul.gca()
            fig = [fig_class, fig_rul]

        self._scatter_class(ax_class, viz_embeddings, logged_labels)
        self._scatter_rul(ax_rul, viz_embeddings, logged_ruls)

        return fig

    def _scatter_class(self, ax, embeddings, labels):
        class_colors = [self.class_cm.colors[c] for c in labels]
        ax.scatter(
            embeddings[:, 0],
            embeddings[:, 1],
            c=class_colors,
            alpha=0.4,
            edgecolors="none",
            s=[4],
            rasterized=True,
        )
        ax.legend(["Source", "Target"], loc="lower left")

    def _scatter_rul(self, ax, embeddings, ruls):
        ax.scatter(
            embeddings[:, 0],
            embeddings[:, 1],
            c=ruls,
            alpha=0.4,
            edgecolors="none",
            s=[4],
            rasterized=True,
        )
        color_bar = plt.cm.ScalarMappable(
            mplcolors.Normalize(ruls.min(), ruls.max()), self.rul_cm
        )
        color_bar.set_array(ruls)
        plt.colorbar(color_bar)


class RMSELoss(torchmetrics.Metric):
    def __init__(self):
        super().__init__()

        self.mse = nn.MSELoss()

        self.add_state("losses", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_elements", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, inputs: torch.Tensor, targets: torch.Tensor):
        summed_square = nn.functional.mse_loss(inputs, targets, reduction="sum")
        self.losses += summed_square
        self.num_elements += inputs.shape[0]

    def compute(self) -> torch.Tensor:
        return torch.sqrt(self.losses / self.num_elements)


class SimpleMetric(torchmetrics.Metric):
    def __init__(self, reduction="mean"):
        super().__init__()

        if reduction not in ["sum", "mean"]:
            raise ValueError(f"Unsupported reduction {reduction}")
        self.reduction = reduction

        self.add_state("losses", default=[], dist_reduce_fx="cat")
        self.add_state("sizes", default=[], dist_reduce_fx="cat")

    def update(self, loss: torch.Tensor, batch_size: int):
        self.losses.append(loss)
        self.sizes.append(torch.tensor(batch_size, device=loss.device))

    def compute(self) -> torch.Tensor:
        if self.reduction == "mean":
            loss = self._weighted_mean()
        else:
            loss = self._sum()

        return loss

    def _weighted_mean(self):
        weights = torch.stack(self.sizes)
        weights = weights / weights.sum()
        loss = torch.stack(self.losses)
        loss = torch.sum(loss * weights)

        return loss

    def _sum(self):
        loss = torch.stack(self.losses)
        loss = torch.sum(loss)

        return loss


class RULScore:
    def __init__(self, pos_factor=10, neg_factor=-13):
        self.pos_factor = pos_factor
        self.neg_factor = neg_factor

    def __call__(self, inputs, targets):
        dist = inputs - targets
        for i, d in enumerate(dist):
            dist[i] = (d / self.neg_factor) if d < 0 else (d / self.pos_factor)
        dist = torch.exp(dist) - 1
        score = dist.sum()

        return score
