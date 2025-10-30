import numpy as np
import pandas as pd
import graph_tool.all as gt
import matplotlib.pyplot as plt

from simulation_with_time_window import make_node_feature_df


def binned_mean_plot(df, x_col, y_col="cumulative_infected", bin_size=None, title=None, plot_log_log=False, reverse_x_axis=False):
    """
    Plot the mean and standard deviation of y_col within bins of x_col.
    """
    data = df[[x_col, y_col]].dropna()

    xmin, xmax = data[x_col].min(), data[x_col].max()
    if xmin <= 0:
        # Avoid issues with log(0)
        xmin = data.loc[data[x_col] > 0, x_col].min()
    if plot_log_log:
        # Define the number of bins based on bin_size (interpreted as a multiplicative factor)
        if bin_size is None:
            num_bins = 50  # fallback default
        else:
            # bin_size acts as an approximate ratio between consecutive bins
            num_bins = int(np.log(xmax / xmin) / np.log(1 + bin_size))
            num_bins = max(5, num_bins)  # prevent too few bins

        bins = np.logspace(np.log10(xmin), np.log10(xmax), num_bins)
    else:
        # Linear binning as before
        if bin_size is None:
            bin_size = (xmax - xmin) / 30
        start = np.floor(xmin / bin_size) * bin_size
        stop = np.ceil(xmax / bin_size) * bin_size + bin_size
        bins = np.arange(start, stop, bin_size)

    cut = pd.cut(data[x_col], bins=bins, include_lowest=True)
    grp = data.groupby(cut, observed=True)[y_col].agg(['mean', 'std', 'count'])

    # Drop empty bins
    grp = grp[grp['count'] > 0]

    centers = np.array([iv.left + (iv.right - iv.left) / 2 for iv in grp.index])
    mean_vals = grp['mean'].to_numpy()
    std_vals = grp['std'].to_numpy()

    agg = pd.DataFrame({
        'center': centers,
        'mean': mean_vals,
        'std': std_vals,
    })

    # --- Plot ---
    plt.figure(figsize=(10, 6))
    plt.plot(agg['center'], agg['mean'], marker='o', linestyle='-', label='Mean')
    plt.fill_between(
        agg['center'],
        agg['mean'] - agg['std'],
        agg['mean'] + agg['std'],
        color='gray',
        alpha=0.3,
        label='±1 Std. Dev.'
    )

    plt.xlabel(x_col)
    plt.ylabel(f'{y_col} (mean ± std)')
    plt.title(title)
    plt.legend()
    if plot_log_log:
        plt.xscale('log')
        plt.yscale('log')

    if reverse_x_axis:
        plt.gca().invert_xaxis()

    plt.tight_layout()
    plt.savefig(f'plots/cumulative_infected_vs_{x_col}.png')
    plt.close()
    # if reverse_x_axis:
    #     plt.gca().invert_xaxis()

def plot_infections_vs_metric(df):
    # ---------- Usage ----------
    # 1) Degree (exact-by-degree or with bin_size=1)

    binned_mean_plot(df, 'degree',
        title='Average cumulative infections vs Degree', plot_log_log=True, reverse_x_axis=True)

    # 2) Leverage (choose a bin size, e.g. 0.05)
    binned_mean_plot(df, 'leverage',
        title='Average cumulative infections vs Leverage')

    # 3) Betweenness (often skewed; pick a sensible bin size, or define custom bins)

    binned_mean_plot(df, 'betweenness',
        title='Average cumulative infections vs Betweenness', plot_log_log=True, reverse_x_axis=True)

    binned_mean_plot(df, "strength", title="Average cumulative infections vs Strength", plot_log_log=True, reverse_x_axis=True)

    binned_mean_plot(df, "betweenness_time", title="Average cumulative infections vs Betweenness time", plot_log_log=True, reverse_x_axis=True)

    binned_mean_plot(df, "wts", title="Average cumulative infections vs Weighted activation")

def main():
    EXPERIMENT_NAME = "sis_sim_max_steps=1000,start=1000,vaccine_strategy=VaccinationStrategy.NONE,vaccine_fraction=0.1,immunity_decay_rate=0.998,use_natural_immunity=False"
    DIR_NAME = f"./cache/{EXPERIMENT_NAME}"
    num_seeds = 10
    all_stats = []
    for i in range(num_seeds):
        path = f"{DIR_NAME}/{i}.gt"
        g = gt.load_graph(path)
        statistics_df = make_node_feature_df(g)
        all_stats.append(statistics_df)
    big_df = pd.concat(all_stats)
    plot_infections_vs_metric(big_df)


if __name__ == "__main__":
    main()