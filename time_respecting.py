import graph_tool.all as gt
import numpy as np
from heapq import heappush, heappop
from tqdm import tqdm
import csv
from collections import Counter


def compute_betweenness_from_csv(csv_file, output_file="betweenness.csv"):
    betweenness = Counter()
    total_paths = 0

    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Processing paths"):
            path = list(map(int, row["path"].split()))
            for node in path[1:-1]:
                betweenness[node] += 1
            total_paths += 1

    for node in betweenness:
        betweenness[node] /= total_paths

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node", "betweenness"])
        for node, val in sorted(betweenness.items()):
            writer.writerow([node, val])

    print(f"Saved betweenness scores to {output_file}")

def compute_avg_time_respecting_path_length(csv_file):
    total_length = 0
    path_count = 0

    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Calculating average path length"):
            hops = int(row["hops"])
            total_length += hops
            path_count += 1

    avg_length = total_length / path_count if path_count > 0 else 0
    print(f"Average time-respecting path length: {avg_length}")
    return avg_length

def compute_avg_shortest_time(csv_file):
    total_time = 0
    path_count = 0

    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Calculating average shortest time"):
            arrival_time = float(row["arrival_time"])
            total_time += arrival_time
            path_count += 1

    avg_time = total_time / path_count if path_count > 0 else 0
    print(f"Average shortest time-respecting path time: {avg_time}")
    return avg_time

def temporal_shortest_paths_with_paths(g, source, start_time=0):
    time_prop = g.ep["time"]
    T = {int(v): np.inf for v in g.vertices()}
    hops = {int(v): np.inf for v in g.vertices()}
    parent = {int(v): None for v in g.vertices()}
    T[int(source)] = start_time
    hops[int(source)] = 0
    pq = [(start_time, 0, int(source))]

    while pq:
        t_u, h_u, u = heappop(pq)
        if t_u > T[u]:
            continue
        for e in g.vertex(u).out_edges():
            v = int(e.target())
            t_e = time_prop[e]
            if t_e >= t_u:
                arrival = t_e
                if arrival < T[v]:
                    T[v] = arrival
                    hops[v] = h_u + 1
                    parent[v] = u
                    heappush(pq, (arrival, h_u + 1, v))
    return T, hops, parent


def reconstruct_path(parent, src, dst):
    path = []
    cur = int(dst)
    while cur is not None:
        path.append(cur)
        if cur == int(src):
            break
        cur = parent[cur]
    if path[-1] != int(src):
        return None
    return list(reversed(path))


def temporal_analysis(g, output_file="temporal_paths.csv"):
    vertices = list(g.vertices())

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "arrival_time", "hops", "path"])

        for src in tqdm(vertices):
            T, hops, parent = temporal_shortest_paths_with_paths(g, src)
            for dst in vertices:
                if src == dst: 
                    continue
                if np.isfinite(T[int(dst)]):
                    path = reconstruct_path(parent, src, dst)
                    if path:
                        writer.writerow([
                            int(src),
                            int(dst),
                            T[int(dst)],
                            hops[int(dst)],
                            " ".join(map(str, path))
                        ])

    print(f"Results saved to {output_file}")


def main():
    g = gt.collection.ns["escorts"]
    # temporal_analysis(g)
    # compute_betweenness_from_csv("temporal_paths.csv", "time_respecting_betweenness.csv")
    # compute_avg_time_respecting_path_length("temporal_paths.csv")
    compute_avg_shortest_time("temporal_paths.csv")


if __name__ == "__main__":
    main()