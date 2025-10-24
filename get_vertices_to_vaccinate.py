import numpy as np
import graph_tool.all as gt
from rich import print
import random
from tqdm import trange
import matplotlib.pyplot as plt
import pandas as pd
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

    return d

def main():
    d = vertices_to_vaccinate_per_strategy(
        g = gt.collection.ns["escorts"],
        start = 1000,
        vaccine_fraction = 0.1,
        max_steps = None
    )

if __name__ == "__main__":
    main()
