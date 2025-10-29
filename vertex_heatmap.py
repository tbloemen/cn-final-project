import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list

def plot_heatmap(df):
    # group vertices by strategy_name
    strategy_names = df.strategy_name.unique()
    sets = []
    for strategy_name in strategy_names:
        df_strategy = df[df.strategy_name == strategy_name]
        vertices = set(df_strategy.vertex_index)
        sets.append((strategy_name, vertices))

    # compute Jaccard similarity matrix
    n = len(strategy_names)
    data = np.zeros((n, n))
    for i, (_, vertices_i) in enumerate(sets):
        for j, (_, vertices_j) in enumerate(sets):
            intersection = vertices_i.intersection(vertices_j)
            union = vertices_i.union(vertices_j)
            data[i, j] = len(intersection) / len(union) if len(union) > 0 else 0

    # hierarchical clustering to reorder strategies
    # use "1 - similarity" as distance
    distance_matrix = 1 - data
    linkage_matrix = linkage(distance_matrix, method="average")
    order = leaves_list(linkage_matrix)

    # reorder data and strategy names
    data = data[order][:, order]
    ordered_names = [strategy_names[i] for i in order]

    # plot heatmap
    plt.figure(figsize=(8, 6))
    im = plt.imshow(data, cmap="hot", interpolation="nearest")

    # colorbar and labels
    plt.colorbar(im, label="Jaccard Similarity")
    plt.title("Vertex Overlap Between Strategies")

    # axis ticks
    plt.xticks(np.arange(len(ordered_names)), ordered_names, rotation=45, ha="right")
    plt.yticks(np.arange(len(ordered_names)), ordered_names)

    plt.tight_layout()
    plt.savefig("plots/vertex_heatmapz.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv("cache/vertices-to-vaccinate/start=1000_frac=0.1_max=None_2025-10-26_14h06m14s.csv")
    plot_heatmap(df)
