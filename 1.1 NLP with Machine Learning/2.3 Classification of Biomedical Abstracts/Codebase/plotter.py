import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.preprocessing import MinMaxScaler
import pandas as pd

from helpers import REPORT_DIR, PLOT_DIR

scaler = MinMaxScaler((1,2))

if __name__ == "__main__":
    # fold:n_components : from 10000 features
    svd_df = pd.read_csv(REPORT_DIR / 'svd_comp_selection_data.tsv', sep='\t')
    trials = [
        svd_df[svd_df['trial'] == 0],
        svd_df[svd_df['trial'] == 1]
    ]
    
    fig = plt.figure(figsize=(8,6))
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    
    ax = fig.add_subplot(221)
    ax1 = fig.add_subplot(222)
    ax2 = fig.add_subplot(223)


    ax.plot(trials[0]['n_components'], trials[0]['exp_var'], 
        c='k', marker='.', linestyle='-', linewidth=2, markersize=12)
    ax.set_ylabel('% of Variance Explained')
    ax.set_xlabel('Num. of Components')


    norm_time = scaler.fit_transform(trials[0]['time'].to_numpy().reshape(-1,1)).flatten()
    ax1.plot(trials[0]['n_components'], norm_time, 
    c='k', marker='.', linestyle='-', linewidth=2, markersize=12)
    ax1.set_ylabel('Time Taken (seconds)')
    ax1.set_xlabel('Num. of Components')

    efficiency = (trials[0]['exp_var']/norm_time)

    ax2.plot(trials[0]['n_components'], efficiency, 
    c='k', marker='.', linestyle='-', linewidth=2, markersize=12)
    ax2.set_ylabel('Efficiency (Var. Expl./Time)')
    ax2.set_xlabel('Num. of Components')

    # ax.set_xscale('symlog')
    plt.show()