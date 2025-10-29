from matplotlib.pylab import Enum
import graph_tool.all as gt
import numpy as np
import matplotlib.pyplot as plt

class VaccinationStrategy(Enum):
    DEGREE = 1
    LEVERAGE = 2
    STRENGTH = 3
    BETWEENNESS = 4
    BETWEENNESS_TIME = 5
    WTS = 6

CACHE_DIR = './cache'
OPTIONS = {
    "max_steps":            1000,
    "start":                1000,
    "vaccine_strategy":     VaccinationStrategy.DEGREE,
    "vaccine_fraction":     0.1,
    "immunity_decay_rate":  0.990,
    "use_natural_immunity": False       # the natural immunity case, sets immunity to 1 if infected
}
EXPERIMENT_NAME: str = "sis_sim_" + ','.join(f'{k}={v}' for k,v in OPTIONS.items())

cache = f'{CACHE_DIR}/{EXPERIMENT_NAME}'

for i in range(10):
    graph_fn = f'{cache}/{i}.gt'
    sim_fn = f'{cache}/{i}.npy'
    #tiot_fn = f'{cache}/tiot_{i}.npy'

    graph = gt.load_graph(graph_fn)
    sim = np.load(sim_fn, allow_pickle=False)
    #tiot = np.load(tiot_fn, allow_pickle=False)

    # plotting the cumulative infected
    cumulative_infected = graph.vertex_properties["cumulative_infected"]
    data = [0]
    for j,ci in enumerate(cumulative_infected):
        data.append(data[j-1]+ci)
    plt.plot(data, label=f'run={i+1}')
    
plt.legend()
plt.xlabel("Time")
plt.ylabel("Cumulative infected")
plt.title("Cumulative Infected over Time")
plt.savefig(f'plots/cumulative_infected_{EXPERIMENT_NAME}.png')