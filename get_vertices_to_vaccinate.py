import numpy as np
import graph_tool.all as gt
from rich import print
import random
from tqdm import trange
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo  # Python 3.9+
from simulation_with_time_window import simulate_SIS, make_node_feature_df, VaccinationStrategy


def vertices_to_vaccinate_per_strategy(
    g: gt.Graph = gt.collection.ns["escorts"],
    start: int = 1000,
    vaccine_fraction: float = 0.1,
    max_steps: int | None = None
) -> dict[VaccinationStrategy, list[tuple[gt.Vertex, float]]]:
    
    """ 
    Note: Still have to add time respecting betweenness and WTS
    
    Returns a dict where the keys are vaccination strategies and the items are a list of
    tuples containing the vertices to vaccinate and the relevant vertex metric for the strategy.
    """
    
    def make_list_to_vaccinate(
        num_to_vaccinate: int,
        metric_values: dict[gt.Vertex, float]
    ) -> list[tuple[gt.Vertex, float]]:
        """
        Returns list of tuples containing vertices to vaccinate and their metric value.
        The list is high to low ordered by metric value  
        """
        ranked_active = sorted(
            ((v, metric_values[v]) for v in active_vertices),
            key=lambda item: item[1],
            reverse=True
        )
        to_vaccinate = ranked_active[:num_to_vaccinate]
        return to_vaccinate

    # make set of active edges given start
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
            
    # number of vertices to vaccinate
    num_to_vaccinate = int(len(active_vertices) * vaccine_fraction)

    # this will become the output dict
    d = {}

    # add DEGREE vertices to vaccinate
    degree = g.degree_property_map("total")
    metric_values = {v: degree[v] for v in g.vertices()}
    d[VaccinationStrategy(1)] = make_list_to_vaccinate(num_to_vaccinate, metric_values)
    print("degree compuation done")

    # add LEVERAGE vertices to vaccinate
    leverage = g.new_vertex_property("float")
    for v in g.vertices():
        deg = v.out_degree() + v.in_degree()
        if deg > 1:
            degree_sum = sum(
                ((deg - (u.out_degree() + u.in_degree())) / (deg + (u.out_degree() + u.in_degree())))
                for u in v.all_neighbors()
            )
            leverage[v] = degree_sum / deg
        else:
            leverage[v] = 0.0

    metric_values = {v: leverage[v] for v in g.vertices()}
    d[VaccinationStrategy(2)] = make_list_to_vaccinate(num_to_vaccinate, metric_values)
    print("leverage compuation done")
        
    # add STRENGTH vertices to vaccinate
    strength = g.new_vertex_property("float")
    ratings = g.edge_properties["rating"]
    for e in g.edges():
        src, trt = e.source(), e.target()
        strength[src] += ratings[e]
        strength[trt] += ratings[e]

    metric_values = {v: strength[v] for v in g.vertices()}
    d[VaccinationStrategy(3)] = make_list_to_vaccinate(num_to_vaccinate, metric_values)
    print("strength compuation done")

    # add BETWEENNESS vertices to vaccinate
    betweenness = gt.betweenness(g)[0]
    metric_values = {v: betweenness[v] for v in g.vertices()}
    d[VaccinationStrategy(4)] = make_list_to_vaccinate(num_to_vaccinate, metric_values)
    print("betweenness compuation done")

    return d, active_vertices

from pathlib import Path
import pandas as pd
import datetime as dt

def save_vertices_to_cache(
    d,
    g,
    start: int,
    vaccine_fraction: float,
    max_steps: int | None,
    out_dir: str = "cache/vertices-to-vaccinate",
    stem: str | None = None,
):
    """
    Save dict[VaccinationStrategy, list[(gt.Vertex, float)]] to a csv in repo.
    Returns the written path.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for strat, items in d.items():
        strat_value = int(strat.value)
        strat_name = strat.name
        for v, metric in items:
            rows.append({
                "strategy": int(strat_value),
                "strategy_name": strat_name,
                "vertex_index": int(g.vertex_index[v]),
                "metric": float(metric),
                "start": int(start),
                "vaccine_fraction": float(vaccine_fraction),
                "max_steps": -1 if max_steps is None else int(max_steps),
                "saved_at_utc": dt.datetime.utcnow().isoformat(timespec="seconds"),
            })

    df = pd.DataFrame(rows)

    # filename with parameters for easy reuse; feel free to adjust
    if stem is None:
        ms = "None" if max_steps is None else str(max_steps)
        now_ams = datetime.now(ZoneInfo("Europe/Amsterdam"))
        ts = now_ams.strftime("%Y-%m-%d_%Hh%Mm%Ss")         # e.g. 2025-10-25_11h55m12s
        stem = f"start={start}_frac={vaccine_fraction}_max={ms}_{ts}"

    path_parquet = out / f"{stem}.parquet"
    try:
        df.to_parquet(path_parquet, index=False)  # requires pyarrow or fastparquet
        return path_parquet
    except Exception:
        # fallback to CSV if parquet isn’t available
        path_csv = out / f"{stem}.csv"
        df.to_csv(path_csv, index=False)
        return path_csv


def load_vertices_from_cache(g, path):
    """
    Load cache file and reconstruct dict[VaccinationStrategy, list[(gt.Vertex, float)]].
    File is saved in the folder cache/vertices-to-vaccinate. 
    So just use "cache/vertices-to-vaccinate/filename" as path.
    """
    if str(path).endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    out = {}
    for _, row in df.iterrows():
        vs = VaccinationStrategy(int(row["strategy"]))
        v = g.vertex(int(row["vertex_index"]))
        out.setdefault(vs, []).append((v, float(row["metric"])))
    return out


def main():
    """ See code below for an example how to compute and save and/or load the vertices to vaccinate"""
    
    print("Hello from get_vertices_to_vaccinate.py!")
    # d, active_vertices  = vertices_to_vaccinate_per_strategy(
    #     g = gt.collection.ns["escorts"],
    #     start = 1000,
    #     vaccine_fraction = 0.1,
    #     max_steps = None
    # )

    # path = save_vertices_to_cache(
    #     d, g=gt.collection.ns["escorts"], start=1000, vaccine_fraction=0.1, max_steps=None
    # )

    # print(f"Saved cache to {path}")

    # later (or in a new session), load:
    filename = "start=1000_frac=0.1_max=None_2025-10-25_12h12m30s.csv"
    path = "cache/vertices-to-vaccinate/" + filename
    d2 = load_vertices_from_cache(gt.collection.ns["escorts"], path)

    for i in range(1, 5):
        vs = VaccinationStrategy(i)
        print(vs)
        print(d2[vs][:5])
        print("-------------------------------------------------------")

if __name__ == "__main__":
    main()
