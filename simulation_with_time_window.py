import numpy as np
import graph_tool.all as gt
from rich import print
import random
from tqdm import trange
import matplotlib.pyplot as plt
import pandas as pd


def simulate_SIS(
    g: gt.Graph,
    start_infection_rate=0.1,
    days_infected=100,
    max_steps=None,
    beta=0.9,
    start=0,
):
    """Simulates SIS epidemic with infections starting at time `start` and infecting for `days_infected` days."""
    edge_time = g.edge_properties["time"]
    max_time = int(max(edge_time[e] for e in g.edges()))
    if max_steps is not None:
        max_time = min(max_time, max_steps + start)

    active_vertices = set()
    for e in g.edges():
        t = edge_time[e]
        if start <= t <= max_time:
            active_vertices.add(e.source())
            active_vertices.add(e.target())

    # 0: susceptible, 1: infected
    state = g.new_vertex_property("int")
    cumulative_infected = g.new_vertex_property("int")
    last_infected = g.new_vertex_property("int")
    activated = g.new_vertex_property("bool")

    for v in g.vertices():
        if v in active_vertices and random.random() < start_infection_rate:
            state[v] = 1
            last_infected[v] = start
            activated[v] = False
            cumulative_infected[v] = 1
        else:
            state[v] = 0
            last_infected[v] = -days_infected
            activated[v] = True
            cumulative_infected[v] = 0

    infected_fraction = []
    range = trange(start, max_time + 1)
    for t in range:
        active_edges = [e for e in g.edges() if edge_time[e] == t]

        # print(active_edges)
        new_state = g.new_vertex_property("int")
        for v in g.vertices():
            new_state[v] = state[v]

        for e in active_edges:
            u, v = e.source(), e.target()
            activated[u] = True
            activated[v] = True

        for v in g.vertices():
            if state[v] > 0 and activated[v]:
                # Recovery step
                new_state[v] = 0 if t - last_infected[v] >= days_infected else 1

        for e in active_edges:
            u, v = e.source(), e.target()
            if state[u] > 0 and state[v] == 0:
                if random.random() < beta:
                    # print(f"Infection from {u} to {v} at time {t}")
                    new_state[v] = 1
                    cumulative_infected[v] += 1
                    last_infected[v] = t
            elif state[v] > 0 and state[u] == 0:
                if random.random() < beta:
                    # print(f"Infection from {v} to {u} at time {t}")
                    new_state[u] = 1
                    cumulative_infected[u] += 1
                    last_infected[u] = t

        state = new_state
        if len(active_vertices) > 0:
            frac = np.mean([(1 if state[v] > 0 else 0) for v in active_vertices])
        else:
            frac = 0
        range.set_postfix({"Infected fraction": frac})
        infected_fraction.append(frac)
    # Link to cumulative infected property for further analysis
    g.vertex_properties["cumulative_infected"] = cumulative_infected

    return infected_fraction, g

def leverage(g: gt.Graph, deg):
    """
    Unweighted leverage centrality for all vertices. Assumes g is undirected.
    """
    # deg_pm = g.degree_property_map("total")
    # deg_arr = deg_pm.a

    L = g.new_vertex_property("double")

    for v in g.vertices():
        i = int(v)
        ki = deg[i]
        if ki == 0:
            L[v] = 0.0
            continue
        s = 0.0
        for u in v.out_neighbors():
            kj = deg[int(u)]
            denom = ki + kj
            if denom != 0:
                s += (ki - kj) / denom
        L[v] = s / ki

    return L

def make_node_feature_df(g: gt.Graph):
    """
    Create a DataFrame with per-node features combining
    cumulative infections and standard centrality metrics.
    """
    # Ensure the property exists
    if "cumulative_infected" not in g.vertex_properties:
        raise ValueError("Graph must have a 'cumulative_infected' vertex property.")

    cumulative_infected = g.vertex_properties["cumulative_infected"]

    # Compute metrics
    print("Computing centrality metrics...")
    deg = g.degree_property_map("total").a
    print("Computed degree")
    lev = leverage(g, deg).a
    print("Computed leverage")
    bet = gt.betweenness(g)[0].a
    print("Computed betweenness")
    # closeness = gt.closeness(g).a
    # print("Computed closeness")
    # eigen = gt.eigenvector(g)[1].a
    # print("Computed eigenvector")


    # Build dataframe
    df = pd.DataFrame(
        {
            "node": [int(v) for v in g.vertices()],
            "degree": deg,
            "leverage": lev,
            "betweenness": bet,
            # "closeness": closeness,
            # "eigenvector": eigen,
            "cumulative_infected": [cumulative_infected[v] for v in g.vertices()],
        }
    )

    return df


def main():
    random.seed(42)
    print("Hello from cn-final-project!")
    g = gt.collection.ns["escorts"]

    sim, g = simulate_SIS(g, max_steps=100, start=1000)

    plt.plot(sim)
    plt.xlabel("Time")
    plt.ylabel("Fraction infected")
    plt.title("Temporal SIS epidemic simulation (undirected)")
    plt.savefig("plots/temporal_sis_simulation_with_time_window.png")
    # plt.show()

    df = make_node_feature_df(g)
    print(df.head())


if __name__ == "__main__":
    main()
