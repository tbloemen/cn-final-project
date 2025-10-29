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
    WEIGHTED_TIME = 6
    WTS = 7


def simulate_SIS(
        g: gt.Graph,
        start_infection_rate: float = 0.1,
        days_infected: int = 100,
        max_steps: int | None = None,
        beta: float = 0.9,
        start: int = 0,
        vaccine_strategy: VaccinationStrategy = None,
        vaccine_fraction: float = 0.1,
        weighted_time_decay_index=0,
        immunity_decay_rate: float = 1.0  # a decay rate of 1.0 means no decay
):
    """Simulates SIS epidemic with infections starting at time `start` and infecting for `days_infected` days."""
    edge_time = g.edge_properties["time"]
    max_time = int(max(edge_time[e] for e in g.edges()))
    if max_steps is not None:
        max_time = min(max_time, max_steps + start)

    num_total_infections = 0
    num_total_infected_over_time = []

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
                ((deg - (u.out_degree() + u.in_degree())) / (deg + (u.out_degree() + u.in_degree()))) for u in
                v.all_neighbors()
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
    state = g.new_vertex_property("int")  # 0: susceptible, 1: infected
    cumulative_infected = g.new_vertex_property("int")  # number of times the vertex has been infected: >= 0
    last_infected = g.new_vertex_property("int")  # timestep at which the vertex has last been infected
    activated = g.new_vertex_property("bool")  # not completely sure what this property does
    vaccinated = g.new_vertex_property("bool")  # whether the vertex has been vaccinated or not
    immunity = g.new_vertex_property("float")  # immunity of node, will decay each t_step by rate

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
        case VaccinationStrategy.WEIGHTED_TIME:
            wt = gt.load_graph("node_rankings_time_weighted_activation.gt")
            # Now build your metric_values dict using vertex indices
            # metric_values = {v: vprop_act[v] for v in g.vertices()}
            # node activation with different decay rates
            # decay rates saved = [0.999, 0,995, 0.99, 0.895, 0.89]
            # get with weighted_activation_node_i
            vprop_act = wt.vertex_properties[f"weighted_activation_node_{weighted_time_decay_index}"]

            metric_values = {v: vprop_act[v] for v in g.vertices()}

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
            activated[v] = False  # this flag is unrelated to vaccination; False is fine
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
        num_total_infections += 1
        activated[v] = False
    num_total_infected_over_time.append(num_total_infections)
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
    print(f"fraction vaccinated at start: {counter / len(active_vertices)}")

    # print faction of infected at start
    counter = 0
    for v in g.vertices():
        if state[v] == 1:
            counter += 1
    print(f"fraction infected at start: {counter / len(active_vertices)}")

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
                    # immunity[v] = 1
                    cumulative_infected[v] += 1
                    last_infected[v] = t
                    num_total_infections += 1
            elif state[v] > 0 and state[u] == 0:
                if random.random() < beta * (1 - immunity[v]):
                    # print(f"Infection from {v} to {u} at time {t}")
                    new_state[u] = 1
                    # immunity[u] = 1
                    cumulative_infected[u] += 1
                    last_infected[u] = t
                    num_total_infections += 1

        state = new_state
        if len(active_vertices) > 0:
            frac = np.mean([(1 if state[v] > 0 else 0) for v in active_vertices])
        else:
            frac = 0
        range.set_postfix({"Infected fraction": frac})
        infected_fraction.append(frac)
        num_total_infected_over_time.append(num_total_infections)

    # Add cumulative_infected, strength and leverage to vertex properties
    g.vertex_properties["cumulative_infected"] = cumulative_infected
    g.vertex_properties["strength"] = strength
    g.vertex_properties["leverage"] = leverage

    return infected_fraction, g, num_total_infected_over_time


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
        "max_steps": 1000,
        "start": 1000,
        "vaccine_strategy": VaccinationStrategy.DEGREE,
        "vaccine_fraction": 0.1,
        "immunity_decay_rate": 0.990
    }
    EXPERIMENT_NAME: str = "sis_sim_" + ','.join(f'{k}={v}' for k, v in OPTIONS.items())

    decay_rates = [0.999, 0.995, 0.99, 0.895, 0.89]
    EXPERIMENT_NAME = "sis_decay_experiment_with_time_activation"
    CACHE_DIR = f'./cache/{EXPERIMENT_NAME}'
    os.makedirs(CACHE_DIR, exist_ok=True)

    # --- Collect results ---
    infections_over_time_per_decay = []

    for i, decay in enumerate(decay_rates):
        infections_per_seed = []
        for index, seed in enumerate(SEEDS):
            print(f'Starting iteration {index + 1}, decay={decay}, seed={seed}')
            random.seed(seed)

            sim, g, infections_over_time = simulate_SIS(
                g_escorts,
                max_steps=1000,
                start=1000,
                vaccine_strategy=VaccinationStrategy.WEIGHTED_TIME,
                vaccine_fraction=0.1,
                immunity_decay_rate=0.990,
                weighted_time_decay_index=i
            )

            infections_per_seed.append(infections_over_time)

            # Save sim and graph for this seed
            print('Saving simulation + graph...')
            os.makedirs(f'{CACHE_DIR}/decay_{i}', exist_ok=True)
            g.save(f'{CACHE_DIR}/decay_{i}/{index}.gt')
            np.save(f'{CACHE_DIR}/decay_{i}/{index}.npy', sim, allow_pickle=False)

        # Save the infections_per_seed for this decay rate to avoid recomputation
        np.save(f'{CACHE_DIR}/infections_decay_{i}.npy', np.array(infections_per_seed, dtype=object))
        infections_over_time_per_decay.append(infections_per_seed)
    print(f'Starting {EXPERIMENT_NAME}')
    decay_rates = [0.999, 0.995, 0.99, 0.895, 0.89]
    infections_over_time_per_decay = []
    # we are going to run this simulation `len(SEEDS)` times, ensuring the same seeds each time project is ran
    for i in range(5):
        infections_per_seed = []
        for index, seed in enumerate(SEEDS):
            print(f'Starting computing iteration {index + 1}, with seed: {seed}')
            random.seed(seed)

            sim, g, infections_over_time = simulate_SIS(g_escorts, max_steps=1000, start=1000,
                                                        vaccine_strategy=VaccinationStrategy.WEIGHTED_TIME,
                                                        vaccine_fraction=0.1, weighted_time_decay_index=i)
            infections_per_seed.append(infections_over_time)
            # saving to cache, can be loaded using `np.load` and `gt.load`
            print('Saving to cache...')
            DIR_NAME = f'./cache/{EXPERIMENT_NAME}'
            os.makedirs(DIR_NAME, exist_ok=True)
            g.save(f'{DIR_NAME}/{index}.gt')
            np.save(f'{DIR_NAME}/{index}.npy', sim, allow_pickle=False)
            results.append((sim, g))
        infections_over_time_per_decay.append(infections_per_seed)

    average_infections = []

    plt.figure(figsize=(10, 6))

    for i, decay in enumerate(decay_rates):
        infections_per_seed = np.load(f'{CACHE_DIR}/infections_decay_{i}.npy', allow_pickle=True)
        infections_per_seed = np.stack(infections_per_seed).astype(float)
        # infections_per_seed = np.array(list(infections_per_seed))  # (num_seeds, num_timesteps)

        mean = infections_per_seed.mean(axis=0)
        std = infections_per_seed.std(axis=0)
        plt.plot(mean, label=f"decay={decay}")
        plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2)

    plt.xlabel("Time step")
    plt.ylabel("Average infections")
    plt.title("Average infections over time for different decay rates")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/average_infections_over_time_per_decay_rate_with_immunity_decay")
    plt.show()


if __name__ == "__main__":
    main()
