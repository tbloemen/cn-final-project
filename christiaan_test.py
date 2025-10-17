import numpy as np
import graph_tool.all as gt
from rich import print
import random
from tqdm import trange
import matplotlib.pyplot as plt


def simulate_SIS(
    g: gt.Graph, start_infection_rate=0.1, days_infected=2000, max_steps=None, beta=0.9
):
    edge_time = g.edge_properties["time"]
    max_time = int(max(edge_time[e] for e in g.edges()))
    if max_steps is not None:
        max_time = min(max_time, max_steps)

    # 0: susceptible, 1: infected, 2: initially infected
    state = g.new_vertex_property("int")
    last_infected = g.new_vertex_property("int")
    activated = g.new_vertex_property("bool")

    for v in g.vertices():
        state[v] = 2 if random.random() < start_infection_rate else 0
        last_infected[v] = 0 if state[v] == 2 else -days_infected
        activated[v] = False if state[v] == 2 else True

    infected_fraction = []
    for t in trange(max_time + 1):
        if t < 1000:
            continue
        active_edges = [e for e in g.edges() if edge_time[e] == t]

        # print(active_edges)
        new_state = g.new_vertex_property("int")

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
        frac = np.mean([state[v] if state[v] < 2 else 1 for v in g.vertices()])
        print(frac)
        infected_fraction.append(frac)
    return infected_fraction


def main():
    print("Hello from cn-final-project!")
    g = gt.collection.ns["escorts"]
    print(g)
    print(g.edge_properties)
    print(g.vertex_properties)
    sim = simulate_SIS(g, max_steps=2000)

    plt.plot(sim)
    plt.xlabel("Time")
    plt.ylabel("Fraction infected")
    plt.title("Temporal SIS epidemic simulation (undirected)")
    plt.savefig("christiaan_test.png")
    plt.show()


if __name__ == "__main__":
    main()
