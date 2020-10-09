import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import umap


class EmbeddingViz(pl.metrics.Metric):
    def __init__(self, num_elements, embedding_size):
        super().__init__()

        self.num_elements = num_elements
        self.embedding_size = embedding_size

        self.add_state('embeddings', default=torch.zeros(num_elements, embedding_size), dist_reduce_fx=None)
        self.add_state('labels', default=torch.zeros(num_elements), dist_reduce_fx=None)
        self.add_state('sample_counter', default=torch.tensor(0), dist_reduce_fx=None)

    def update(self, embeddings, labels):
        start = self.sample_counter
        end = self.sample_counter + embeddings.shape[0]
        self.sample_counter = end

        self.embeddings[start:end] = embeddings
        self.labels[start:end] = labels

    def compute(self):
        logged_embeddings = self.embeddings[:self.sample_counter].detach().cpu().numpy()
        logged_labels = self.labels[:self.sample_counter].int()

        viz_embeddings = umap.UMAP().fit_transform(logged_embeddings)
        colors = [plt.get_cmap('tab10').colors[c] for c in logged_labels]
        fig = plt.figure(figsize=(10, 10))
        plt.scatter(viz_embeddings[:, 0], viz_embeddings[:, 1], c=colors)

        return fig
