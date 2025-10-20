import numpy as np
import graph_tool.all as gt
from rich import print
import random
from tqdm import trange
import matplotlib.pyplot as plt


def simulate_SIS(
    g: gt.Graph, start_infection_rate=0.1, days_infected=100, max_steps=None, beta=1, start=0, use_immunity=False, immunity_decay=0.99894394847883
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
    immunity = g.new_vertex_property("float")
    last_infected = g.new_vertex_property("int")
    activated = g.new_vertex_property("bool")
    ever_infected = g.new_vertex_property("int")

    for v in g.vertices():
        if v in active_vertices and random.random() < start_infection_rate:
            state[v] = 2
            last_infected[v] = start
            immunity[v] = 1
            activated[v] = False
        else:
            state[v] = 0
            last_infected[v] = -days_infected
            immunity[v] = 0
            activated[v] = True

    infected_fraction = []
    average_immunity = []
    num_infected = 0
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
            # immunity decays by 68% in 1 year
            immunity[v] = immunity[v] * immunity_decay
            if state[v] > 0 and activated[v]:
                # Recovery step
                new_state[v] = 0 if t - last_infected[v] >= days_infected else 1

        for e in active_edges:
            u, v = e.source(), e.target()
            if state[u] > 0 and state[v] == 0:
                if random.random() < beta * (1-immunity[v] * int(use_immunity)):
                    # print(f"Infection from {u} to {v} at time {t}")
                    new_state[v] = 1
                    immunity[v] = 1
                    last_infected[v] = t
                    ever_infected[v] = 1
                    num_infected += 1
            elif state[v] > 0 and state[u] == 0:
                if random.random() < beta * (1-immunity[v] * int(use_immunity)):
                    # print(f"Infection from {v} to {u} at time {t}")
                    new_state[u] = 1
                    immunity[u] = 1
                    last_infected[u] = t
                    ever_infected[u] = 1
                    num_infected += 1

        state = new_state
        if len(active_vertices) > 0:
            frac = np.mean([(1 if state[v] > 0 else 0) for v in active_vertices])
        else:
            frac = 0
        print(frac)
        average_immunity.append(beta * (1 - np.mean([immunity[v] for v in g.vertices()])))
        infected_fraction.append(frac)
        print("Fraction of nodes ever infected:" + str(np.mean(ever_infected)))
    return infected_fraction, average_immunity


def main():
    random.seed(42)
    print("Hello from cn-final-project!")
    g = gt.collection.ns["escorts"]
    print(g)
    print(g.edge_properties)
    print(g.vertex_properties)
    decayrates = [0.990]
    infected_fraction=[]
    average_immunity=[]
    sim = None
    for i in range(len(decayrates)):
        sim = simulate_SIS(g, max_steps=1000, start=1000, use_immunity=True, immunity_decay=decayrates[i])
        infected_fraction.append(sim[0])
        average_immunity.append(sim[1])
        sim = sim[0]

    plt.plot(sim)
    plt.xlabel("Time")
    plt.ylabel("Fraction infected")
    plt.title("Temporal SIS epidemic simulation (undirected) with immunity")
    plt.savefig("plots/temporal_sis_simulation_with_time_window.png")
    plt.show()
    print(average_immunity, infected_fraction)
    time = np.arange(0, 1001)
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Left axis: infected fraction
    for i in range(len(infected_fraction)):
        decay = decayrates[i]
        ax1.plot(time, infected_fraction[i], label=f"Infected (decay={decay})")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Infected fraction")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Right axis: average immunity
    ax2 = ax1.twinx()
    for i in range(len(infected_fraction)):
        decay = decayrates[i]
        ax2.plot(time, average_immunity[i], linestyle="--", label=f"Effective beta (decay={decay})")

    ax2.set_ylabel("Average immunity")

    # Merge legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="center right")

    plt.title("Infected Fraction and Effective Beta (beta * (1 - avg(immunity))")
    plt.savefig("plots/infected_fraction_and_beta.png")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
