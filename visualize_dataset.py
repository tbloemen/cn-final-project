import numpy as np
import graph_tool.all as gt
from rich import print
import random
from tqdm import trange
import matplotlib.pyplot as plt

def activity_over_time(g: gt.Graph):
    edge_time = g.edge_properties["time"]
    max_time = int(max(edge_time[e] for e in g.edges()))

    activity = []
    for t in trange(max_time + 1):
        active_edges = [e for e in g.edges() if edge_time[e] == t]
        activity.append(len(active_edges))
        print(t, len(active_edges))
    return activity


def main():
    g = gt.collection.ns["escorts"]
    activity = activity_over_time(g)

    plt.plot(activity)
    plt.xlabel("Time")
    plt.ylabel("Number of active edges")
    plt.title("Network Activity Over Time")
    plt.savefig("plots/network_activity.png")
    plt.show()


if __name__ == "__main__":
    main()