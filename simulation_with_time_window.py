from unittest import case
from matplotlib.pylab import Enum
import numpy as np
import graph_tool.all as gt
from rich import print
import random
from tqdm import trange
import matplotlib.pyplot as plt
import pandas as pd
import os

SEEDS: list[int] = [
    94436110,
    45527262,
    46019711,
    3836817,
    66875855,
    67067660,
    44576945,
    12023309,
    85269182,
    43678282
]

class VaccinationStrategy(Enum):
    DEGREE = 1
    LEVERAGE = 2
    STRENGTH = 3
    BETWEENNESS = 4
    BETWEENNESS_TIME = 5
    WTS = 6


def simulate_SIS(
    g: gt.Graph,
    start_infection_rate: float = 0.1,
    days_infected: int = 100,
    max_steps: int | None = None,
    beta: float = 0.9,
    start: int = 0,
    vaccine_strategy: VaccinationStrategy = None,
    vaccine_fraction: float = 0.1,
    immunity_decay_rate: float = 1.0                # a decay rate of 1.0 means no decay
):
    """Simulates SIS epidemic with infections starting at time `start` and infecting for `days_infected` days."""
    edge_time = g.edge_properties["time"]
    max_time = int(max(edge_time[e] for e in g.edges()))
    if max_steps is not None:
        max_time = min(max_time, max_steps + start)

    # Compute strength based on ratings
    strength = g.new_vertex_property("float")
    ratings = g.edge_properties["rating"]
    for e in g.edges():
        src, trt = e.source(), e.target()
        strength[src] += ratings[e]
        strength[trt] += ratings[e]
    
    # print(strength.get_array())

    # Compute leverage
    leverage = g.new_vertex_property("float")
    for v in g.vertices():
        deg = v.out_degree() + v.in_degree() 
        if deg > 1:
            degree_sum = sum(
                ((deg - (u.out_degree() + u.in_degree())) / (deg + (u.out_degree() + u.in_degree()))) for u in v.all_neighbors()
            )
            leverage[v] = degree_sum / deg
        else:
            leverage[v] = 0.0

    # Gather all vertices that are active during time window        
    active_vertices = set()
    for e in g.edges():
        t = edge_time[e]
        if start <= t <= max_time:
            active_vertices.add(e.source())
            active_vertices.add(e.target())

    # Create vertex properties for simulation
    state = g.new_vertex_property("int")                   # 0: susceptible, 1: infected
    cumulative_infected = g.new_vertex_property("int")     # number of times the vertex has been infected: >= 0
    last_infected = g.new_vertex_property("int")           # timestep at which the vertex has last been infected
    activated = g.new_vertex_property("bool")              # not completely sure what this property does
    vaccinated = g.new_vertex_property("bool")             # whether the vertex has been vaccinated or not 
    immunity = g.new_vertex_property("float")               # immunity of node, will decay each t_step by rate

    # Apply vaccination strategy
    metric_values = {}

    match vaccine_strategy:
        case VaccinationStrategy.DEGREE:
            degree = g.degree_property_map("total")
            metric_values = {v: degree[v] for v in g.vertices()}
        case VaccinationStrategy.STRENGTH:
            metric_values = {v: strength[v] for v in g.vertices()}
        case VaccinationStrategy.BETWEENNESS:
            betweenness = gt.betweenness(g)[0]
            metric_values = {v: betweenness[v] for v in g.vertices()}
        case VaccinationStrategy.LEVERAGE:
            metric_values = {v: leverage[v] for v in g.vertices()}
        case _:
            metric_values = {}

    if metric_values:
        num_to_vaccinate = int(len(active_vertices) * vaccine_fraction)
        ranked_active = sorted(
            ((v, metric_values[v]) for v in active_vertices),
            key=lambda item: item[1],
            reverse=True
        )
        to_vaccinate = [v for v, _ in ranked_active[:num_to_vaccinate]] 
        for v in to_vaccinate:
            vaccinated[v] = True
            state[v] = 0
            last_infected[v] = -days_infected
            activated[v] = False     # this flag is unrelated to vaccination; False is fine
            cumulative_infected[v] = 0
            immunity[v] = 1.0

    # infect 10% of the active vertices - only infect unvaccinated vertices
    num_to_infect = int(len(active_vertices) * start_infection_rate)
    unvaccinated_active = [v for v in active_vertices if not vaccinated[v]]
    random.shuffle(unvaccinated_active)
    to_infect = unvaccinated_active[:num_to_infect]  
    for v in to_infect:
        state[v] = 1
        last_infected[v] = start
        cumulative_infected[v] = 1
        activated[v] = False

    # initialize the rest of *active* unvaccinated to clean susceptible
    for v in set(unvaccinated_active) - set(to_infect):
        state[v] = 0
        last_infected[v] = -days_infected
        cumulative_infected[v] = 0
        activated[v] = True
    
    # we have se the `immunity` of vaccinated to 1, but unvaccinated not yet to 0:
    for v in set(unvaccinated_active):
        immunity[v] = 0

    # print fraction of vaccinated at start
    counter = 0
    for v in g.vertices():
        if vaccinated[v] == True:
            counter += 1
    print(f"fraction vaccinated at start: {counter/len(active_vertices)}")

    # print faction of infected at start
    counter = 0
    for v in g.vertices():
        if state[v] == 1:
            counter += 1
    print(f"fraction infected at start: {counter/len(active_vertices)}")

    # from here we start running the actual simulation
    infected_fraction = []
    range = trange(start, max_time + 1)
    for t in range:
        active_edges = [e for e in g.edges() if edge_time[e] == t]

        # print(active_edges)
        new_state = g.new_vertex_property("int")
        for v in g.vertices():
            new_state[v] = state[v]

        # Mark activated nodes
        for e in active_edges:
            u, v = e.source(), e.target()
            activated[u] = True
            activated[v] = True

        for v in g.vertices():
            immunity[v] = immunity[v] * immunity_decay_rate
            if state[v] > 0 and activated[v]:
                # Recovery step
                new_state[v] = 0 if t - last_infected[v] >= days_infected else 1

        for e in active_edges:
            u, v = e.source(), e.target()
            if state[u] > 0 and state[v] == 0:
                if random.random() < beta * (1 - immunity[v]):
                    # print(f"Infection from {u} to {v} at time {t}")
                    new_state[v] = 1
                    immunity[v] = 1
                    cumulative_infected[v] += 1
                    last_infected[v] = t
            elif state[v] > 0 and state[u] == 0:
                if random.random() < beta * (1 - immunity[v]):
                    # print(f"Infection from {v} to {u} at time {t}")
                    new_state[u] = 1
                    immunity[u] = 1
                    cumulative_infected[u] += 1
                    last_infected[u] = t

        state = new_state
        if len(active_vertices) > 0:
            frac = np.mean([(1 if state[v] > 0 else 0) for v in active_vertices])
        else:
            frac = 0
        range.set_postfix({"Infected fraction": frac})
        infected_fraction.append(frac)

    # Add cumulative_infected, strength and leverage to vertex properties
    g.vertex_properties["cumulative_infected"] = cumulative_infected
    g.vertex_properties["strength"] = strength
    g.vertex_properties["leverage"] = leverage

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
    # lev = leverage(g, deg).a
    lev = g.vertex_properties["leverage"]
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
    print("Hello from cn-final-project!")
    g_escorts = gt.collection.ns["escorts"]
    results: list[tuple] = []

    OPTIONS = {
        "max_steps":            1000,
        "start":                1000,
        "vaccine_strategy":     VaccinationStrategy.DEGREE,
        "vaccine_fraction":     0.1,
        "immunity_decay_rate":  0.990
    }
    EXPERIMENT_NAME: str = "sis_sim_" + ','.join(f'{k}={v}' for k,v in OPTIONS.items())

    print(f'Starting {EXPERIMENT_NAME}')
    # we are going to run this simulation `len(SEEDS)` times, ensuring the same seeds each time project is ran
    for index, seed in enumerate(SEEDS):
        print(f'Starting computing iteration {index+1}, with seed: {seed}')
        random.seed(seed)

        sim, g = simulate_SIS(g_escorts, 
                              **OPTIONS)

        # saving to cache, can be loaded using `np.load` and `gt.load`
        print('Saving to cache...')
        DIR_NAME=f'./cache/{EXPERIMENT_NAME}'
        os.makedirs(DIR_NAME, exist_ok=True)
        g.save(f'{DIR_NAME}/{index}.gt')
        np.save(f'{DIR_NAME}/{index}.npy', sim, allow_pickle=False)

        results.append((sim, g))

    # plotting the different runs:
    for iteration,(sim,g) in enumerate(results):
        # df = make_node_feature_df(g)
        plt.plot(sim, label=f'run {iteration+1}')

    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Fraction infected")
    plt.title("Temporal SIS epidemic simulation (undirected)")
    plt.savefig(f"plots/{EXPERIMENT_NAME}.png")
    # plt.show()

    # not sure what this was for
    df = make_node_feature_df(results[0][1])
    print((df.sort_values("degree", ascending=False)).head(20))


if __name__ == "__main__":
    main()
