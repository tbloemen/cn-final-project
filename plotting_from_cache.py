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
    RANDOM = 7
    NONE = 8

CACHE_DIR = './cache'
OPTIONS = {
    "max_steps":            1000,
    "start":                1000,
    "vaccine_strategy":     VaccinationStrategy.RANDOM,
    "vaccine_fraction":     0.1,
    "immunity_decay_rate":  0.998,
    "use_natural_immunity": True       # the natural immunity case, sets immunity to 1 if infected
}

plt.figure(figsize=(8,5))
for strat in VaccinationStrategy:
    OPTIONS["vaccine_strategy"] = strat
    EXPERIMENT_NAME: str = "sis_sim_" + ','.join(f'{k}={v}' for k,v in OPTIONS.items())
    cache = f'{CACHE_DIR}/{EXPERIMENT_NAME}'
    tiots = []
    for i in range(10):
        graph_fn = f'{cache}/{i}.gt'
        sim_fn = f'{cache}/{i}.npy'
        tiot_fn = f'{cache}/tiot_{i}.npy'

        #graph = gt.load_graph(graph_fn)
        #sim = np.load(sim_fn, allow_pickle=False)
        tiot = np.load(tiot_fn, allow_pickle=False)
        
        tiots.append(tiot)

    # compute average tiot
    average_tiot = [sum(x)/len(x) for x in zip(*tiots)]

    # plotting the cumulative infected
    plt.plot(average_tiot, label=f'strategy={str(strat).removeprefix("VaccinationStrategy.")}')
    
plt.legend()
plt.xlabel("Time")
plt.ylabel("Cumulative infected")
plt.title(f"Cumulative Infected over Time per Vaccination Strategy With{'' if OPTIONS["use_natural_immunity"] else 'out'} Natural Immunity")
plt.tight_layout()
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig(f'plots/cumulative_infected_all_strategies_combined{'_with_nat_immunity' if OPTIONS["use_natural_immunity"] else ''}.png')