import numpy as np
import graph_tool.all as gt
from rich import print
import random
from tqdm import trange
import matplotlib.pyplot as plt


def simulate_SIS(
    g: gt.Graph, start_infection_rate=0.1, days_infected=100, max_steps=None, beta=0.9, start=0
):
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

    # 0: susceptible, 1: infected, 2: initially infected
    state = g.new_vertex_property("int")
    last_infected = g.new_vertex_property("int")
    activated = g.new_vertex_property("bool")

    for v in g.vertices():
        if v in active_vertices and random.random() < start_infection_rate:
            state[v] = 2
            last_infected[v] = start
            activated[v] = False
        else:
            state[v] = 0
            last_infected[v] = -days_infected
            activated[v] = True

    infected_fraction = []
    for t in trange(start, max_time + 1):
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
                    last_infected[v] = t
            elif state[v] > 0 and state[u] == 0:
                if random.random() < beta:
                    # print(f"Infection from {v} to {u} at time {t}")
                    new_state[u] = 1
                    last_infected[u] = t

        state = new_state
        if len(active_vertices) > 0:
            frac = np.mean([(1 if state[v] > 0 else 0) for v in active_vertices])
        else:
            frac = 0
        print(frac)
        infected_fraction.append(frac)
    return infected_fraction


def main():
    random.seed(42)
    print("Hello from cn-final-project!")
    g = gt.collection.ns["escorts"]
    print(g)
    print(g.edge_properties)
    print(g.vertex_properties)
    sim = simulate_SIS(g, max_steps=1000, start=1000)

    plt.plot(sim)
    plt.xlabel("Time")
    plt.ylabel("Fraction infected")
    plt.title("Temporal SIS epidemic simulation (undirected)")
    plt.savefig("temporal_sis_simulation_with_time_window.png")
    plt.show()


if __name__ == "__main__":
    main()
