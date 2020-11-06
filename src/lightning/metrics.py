import matplotlib.colors as mplcolors
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import umap
from torch import nn as nn


class EmbeddingViz(pl.metrics.Metric):
    def __init__(self, num_elements, embedding_size):
        super().__init__()

        self.num_elements = num_elements
        self.embedding_size = embedding_size

        self.add_state('embeddings', default=torch.zeros(num_elements, embedding_size), dist_reduce_fx=None)
        self.add_state('labels', default=torch.zeros(num_elements), dist_reduce_fx=None)
        self.add_state('ruls', default=torch.zeros(num_elements), dist_reduce_fx=None)
        self.add_state('sample_counter', default=torch.tensor(0), dist_reduce_fx=None)

        self.class_cm = plt.get_cmap('tab10')
        self.rul_cm = plt.get_cmap('viridis')

    def update(self, embeddings, labels, ruls):
        """
        Add embedding, labels and RUL to the metric.

        :param embeddings: hidden layer embedding of the data points
        :param labels: domain labels with 0 being source and 1 being target
        :param ruls: Remaining Useful Lifetime values of the data points
        """
        start = self.sample_counter
        end = self.sample_counter + embeddings.shape[0]
        self.sample_counter = end

        self.embeddings[start:end] = embeddings
        self.labels[start:end] = labels
        self.ruls[start:end] = ruls

    def compute(self):
        """Compute UMAP and plot points to 2d scatter plot."""
        logged_embeddings = self.embeddings[:self.sample_counter].detach().cpu().numpy()
        logged_labels = self.labels[:self.sample_counter].detach().cpu().int()
        logged_ruls = self.ruls[:self.sample_counter].detach().cpu()
        viz_embeddings = umap.UMAP(random_state=42).fit_transform(logged_embeddings)

        fig, (ax_class, ax_rul) = plt.subplots(1, 2, sharex='all', sharey='all', figsize=(20, 10))

        class_colors = [self.class_cm.colors[c] for c in logged_labels]
        ax_class.scatter(viz_embeddings[:, 0], viz_embeddings[:, 1],
                         c=class_colors, alpha=0.4, edgecolors='none', s=[4])
        ax_class.legend(['Source', 'Target'], loc='lower left')

        ax_rul.scatter(viz_embeddings[:, 0], viz_embeddings[:, 1],
                       c=logged_ruls, alpha=0.4, edgecolors='none', s=[4])
        color_bar = plt.cm.ScalarMappable(mplcolors.Normalize(logged_ruls.min(), logged_ruls.max()), self.rul_cm)
        color_bar.set_array(logged_ruls)
        plt.colorbar(color_bar)

        return fig


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.mse = nn.MSELoss()

    def forward(self, inputs, targets):
        return torch.sqrt(self.mse(inputs, targets))