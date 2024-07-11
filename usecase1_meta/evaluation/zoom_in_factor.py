import math
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt

from evaluation.downstream_task import downstream_task
from evaluation.run_inference import impute_data, load_model
from datasets.preprocessor import generate_datasets_pipeline

def zoom_in_factor(config, rack_data_test, rackdata_len, ingressBytes_max):
    timing = False
    # Load model for zoom_in_factor 25
    print('zoom_in_factor 25')
    model25 = load_model(config, 'checkpoints/coarse25.torch', d_model=40, n_heads=config.n_heads, dim_feedforward=config.dim_feedforward, 
                                zoom_in_factor=25, window_size=1000)
    model25.eval()
    window_size=1000
    window_skip=300
    coarse=25
    test_dataset25 = generate_datasets_pipeline(rack_data_test, window_size,
        window_skip,
        coarse,
        output_coarsening_factor = 1)
    res_pred_z2n_coarse25, res_true_z2n_coarse25, _ = impute_data(config, model25, test_dataset25, rackdata_len, ingressBytes_max, timing,
                            window_size, window_skip, coarse)
    coarse_25 = downstream_task(res_pred_z2n_coarse25, res_true_z2n_coarse25, rackdata_len, ingressBytes_max)
    
    # Load model for zoom_in_factor 50
    print('zoom_in_factor 50')
    model50 = load_model(config, config.z2n_model_dir, d_model=40, n_heads=config.n_heads, dim_feedforward=config.dim_feedforward, 
                            zoom_in_factor=config.zoom_in_factor, window_size=config.window_size)
    model50.eval()
    window_size=1000
    window_skip=100
    coarse=50
    test_dataset50 = generate_datasets_pipeline(rack_data_test, window_size,
        window_skip,
        coarse,
        output_coarsening_factor = 1)
    res_pred_z2n_coarse50, res_true_z2n_coarse50, _ = impute_data(config, model50, test_dataset50, rackdata_len, ingressBytes_max, timing,
                            config.window_size, config.window_skip, config.zoom_in_factor)
    coarse_50 = downstream_task(res_pred_z2n_coarse50, res_true_z2n_coarse50, rackdata_len, ingressBytes_max)
    
    # Load model for zoom_in_factor 100
    print('zoom_in_factor 100')
    model100 = load_model(config, 'checkpoints/coarse100.torch', d_model=10, n_heads=2, dim_feedforward=10, zoom_in_factor=100,\
                            window_size=1000)
    model100.eval()
    window_size=1000
    window_skip=300
    coarse=100
    test_dataset100 = generate_datasets_pipeline(rack_data_test, window_size,
        window_skip,
        coarse,
        output_coarsening_factor = 1)
    res_pred_z2n_coarse100, res_true_z2n_coarse100, _ = impute_data(config, model100, test_dataset100, rackdata_len, ingressBytes_max, timing,\
                            window_size, window_skip, coarse)
    coarse_100 = downstream_task(res_pred_z2n_coarse100, res_true_z2n_coarse100, rackdata_len, ingressBytes_max)

    a = np.stack((coarse_25['MSE'], coarse_25['emd'], coarse_25['Autocorrelation'], coarse_25['99_percentile'])) # (3, 10)
    b = np.stack((coarse_50['MSE'], coarse_50['emd'], coarse_50['Autocorrelation'], coarse_50['99_percentile'])) # (3, 10)
    c = np.stack((coarse_100['MSE'], coarse_100['emd'], coarse_100['Autocorrelation'], coarse_100['99_percentile'])) # (3, 10)

    all_methods = [a,b,c]
    # print(all_methods)

    stats = ['MSE', 'EMD', 'Auto\ncorrelation', '99p']
    methods = ['25X', '50X', '100X']
    fig, ax = plt.subplots(1, 1, figsize=(3,3))
    cmap = plt.colormaps.get_cmap('Blues')
    a = None
    diff = 0
    maxes = np.zeros(4)
    for i in range(len(all_methods)):
        for j in range(4):
            if max(all_methods[i][j]) > maxes[j]:
                maxes[j] = max(all_methods[i][j])
    means = np.zeros(4)
    for i in range(len(all_methods)):
        for j in range(4):
            if np.average(all_methods[i][j]) > means[j]:
                means[j] = np.average(all_methods[i][j])
    for i in range(3):
        r = all_methods[i].copy()
        for j in range(4):
            r[j] /= means[j]
        x = np.array([0,3,6,9])
        width = 0.8
        # print(r)
        mean = np.mean(r,axis=1)
        std = np.std(r,axis=1)
        err_lo = mean - np.min(r,axis=1)
        err_hi = np.max(r,axis=1) - mean
        above_threshold = 0
        below_threshold = mean
        # print(methods[i], mean)
        ax.barh((x+(i-1)* width), below_threshold, width, label = methods[i],\
            error_kw=dict(lw=1, capsize=1, capthick=1),capsize=2, color=cmap(i*50), edgecolor='k')
        # bars = ax.barh((x+(i-1)* width), r, width, label = methods[i],\
        #     error_kw=dict(lw=1, capsize=1, capthick=1),capsize=2, color=cmap(i*50), edgecolor='k')
        # ax.bar_label(bars, fmt='{:,.4f}')
    ax.set_yticks(x)
    ax.set_yticklabels(stats, fontsize=11)
    ax.legend(fontsize=10, ncol=1, loc='upper right')
    ax.grid(linestyle='--', axis='x')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=12)

    a = np.stack((coarse_25['Burst_start_pos'], coarse_25['Burst_height'], coarse_25['Burst_freq'], \
    coarse_25['Burst_duration'], coarse_25['Burst_volume'], coarse_25['IngressAfterBurst'], coarse_25['Total_ingress']))
    b = np.stack((coarse_50['Burst_start_pos'], coarse_50['Burst_height'], coarse_50['Burst_freq'], \
        coarse_50['Burst_duration'], coarse_50['Burst_volume'], coarse_50['IngressAfterBurst'], coarse_50['Total_ingress']))
    c = np.stack((coarse_100['Burst_start_pos'], coarse_100['Burst_height'], coarse_100['Burst_freq'], \
        coarse_100['Burst_duration'], coarse_100['Burst_volume'], coarse_100['IngressAfterBurst'], coarse_100['Total_ingress']))
    all_methods = [a,b,c]
    stats = ['Burst_start_pos', 'Burst_height', 'Burst_freq', 'Burst_duration', \
         'Burst_volume', 'Volume\nAfterBurst', 'Total ingress']
    methods = ['25X', '50X', '100X']
    fig, ax = plt.subplots(1, 1, figsize=(3,3))
    cmap = plt.colormaps.get_cmap('Blues')
    a = None
    diff = 0
    maxes = np.zeros(7)
    for i in range(len(all_methods)):
        for j in range(7):
            if max(all_methods[i][j]) > maxes[j]:
                maxes[j] = max(all_methods[i][j])
    means = np.zeros(7)
    for i in range(len(all_methods)):
        for j in range(7):
            if np.average(all_methods[i][j]) > means[j]:
                means[j] = np.average(all_methods[i][j])

    for i in range(3):
        r = all_methods[i].copy()
        for j in range(7):
            r[j] /= means[j]*1.1
        x = np.arange(0,19,3)
        width = 0.7
        mean = np.mean(r,axis=1)
        std = np.std(r,axis=1)
        err_lo = mean - np.min(r,axis=1)
        err_hi = np.max(r,axis=1) - mean
        above_threshold = 0
        below_threshold = mean
        # print(methods[i], mean)
        ax.barh((x+(i-0.5)* width), below_threshold, width, label = methods[i],\
            error_kw=dict(lw=1, capsize=1, capthick=1),capsize=2, color=cmap(i*50), edgecolor='k')
        # print(r)
        # bars = ax.barh((x+(i-0.5)* width), r, width, label = methods[i],\
        #     error_kw=dict(lw=1, capsize=1, capthick=1),capsize=2, color=cmap(i*50), edgecolor='k')
    ax.set_yticks(x+0.25)
    ax.set_yticklabels(stats)
    ax.legend(fontsize=10, ncol=1, loc='lower right')
    ax.grid(linestyle='--', axis='y')
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)