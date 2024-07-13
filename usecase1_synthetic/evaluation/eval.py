import math
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import torch

# Project modules
from evaluation.downstream_task import downstream_task
from evaluation.run_inference import impute_data, load_model
from evaluation.baselines import knn, plain_transformer, iter_imputer, brits
from model_training.transformer import TSTransformerEncoder
from evaluation.get_certainty_score import pred_withconf

def run_downstream_task(config, test_dataset, train_dataset, rackdata_len, throughput_threshold, res_queue_max):
    res_brits = brits(config, test_dataset, train_dataset, rackdata_len, throughput_threshold)
    print('brits')
    timing = False
    model = load_model(config, config.z2n_model_dir, d_model=config.d_model, n_heads=config.n_heads, dim_feedforward=config.dim_feedforward, 
                                max_len=42, zoom_in_factor=config.zoom_in_factor, window_size=config.window_size)
    model.eval()
    res_pred_z2n, res_true_z2n, _ = impute_data(config, model, test_dataset, rackdata_len, res_queue_max, timing,\
                                    config.window_size, config.window_skip, config.zoom_in_factor)
    print(f"res_pred_z2n: {res_pred_z2n.shape}, res_true_z2n: {res_true_z2n.shape}")
    res_z2n = downstream_task(res_pred_z2n, res_true_z2n, rackdata_len, throughput_threshold)
    print('z2n')

    res_knn = knn(config, test_dataset, train_dataset, rackdata_len, throughput_threshold)
    print('knn')

    res_plain = plain_transformer(config, test_dataset, rackdata_len, throughput_threshold)
    print('plain')

    res_iter = iter_imputer(config, test_dataset, rackdata_len, throughput_threshold)
    print('iter')
    plot(res_z2n, res_knn, res_iter, res_plain, res_brits)

def run_timing(config, test_dataset, rackdata_len, res_queue_max):
    timing = True
    model = load_model(config, config.z2n_model_dir, d_model=config.d_model, n_heads=config.n_heads, dim_feedforward=config.dim_feedforward, 
                                max_len=42, zoom_in_factor=config.zoom_in_factor, window_size=config.window_size)
    model.eval()
    _, _, time_spend = impute_data(config, model, test_dataset, rackdata_len, res_queue_max, timing, \
                                config.window_size, config.window_skip, config.zoom_in_factor)
    print(sum(time_spend) / len(time_spend))

def run_uncertainty(config, test_dataset):
    timing = True
    model = load_model(config, config.z2n_model_dir, d_model=config.d_model, n_heads=config.n_heads, dim_feedforward=config.dim_feedforward, 
                                max_len=42, zoom_in_factor=config.zoom_in_factor, window_size=config.window_size)
    model.eval()
    conf = [] 
    for i in range(len(test_dataset)):
        res_dropout = pred_withconf(model, test_dataset, i, config.window_size, config.zoom_in_factor, T=100)
        res_dropout = np.array(res_dropout)
        res_mc_std = res_dropout.std(axis=0)
        conf.append(np.max(res_mc_std))
    print(sum(conf)/len(conf))
        

def plot(res_z2n, res_knn, res_iter, res_plain, res_brits):
  
    a = np.stack((res_z2n['MSE'], res_z2n['emd'], res_z2n['Autocorrelation'], res_z2n['99_percentile'])) # (3, 10)
    b = np.stack((res_knn['MSE'], res_knn['emd'], res_knn['Autocorrelation'], res_knn['99_percentile']))
    c = np.stack((res_iter['MSE'], res_iter['emd'], res_iter['Autocorrelation'], res_iter['99_percentile']))
    d = np.stack((res_plain['MSE'], res_plain['emd'], res_plain['Autocorrelation'], res_plain['99_percentile']))
    e = np.stack((res_brits['MSE'], res_brits['emd'], res_brits['Autocorrelation'], res_brits['99_percentile']))
    
    all_methods = [a,b,c,d,e]

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

    stats = ['MSE', 'EMD', 'Autocorrelation', '99percentile']
    methods = ['Zoom2Net', 'KNN', 'CoarseGrained', 'PlainTransformer', 'Brits']
    fig, ax = plt.subplots(1, 1, figsize=(6,3))
    cmap = plt.colormaps.get_cmap('Blues')
    diff = 0
    for i in range(5):
        r = all_methods[i].copy()
        for j in range(4):
            r[j] /= means[j] * 1.1 
        if i !=2:
            r[3] = r[3]*30
        x = np.array([0,2,4,6])
        width = 0.3
        mean = np.mean(r,axis=1)
        if i == 0:
            pass
        else:
            diff += sum(mean[1:4] - a)/3
        err_lo = mean - np.min(r,axis=1)
        err_hi = np.max(r,axis=1) - mean
        above_threshold = 0
        below_threshold = mean
        ax.bar((x+(i-2)* width), below_threshold, width, label = methods[i],\
            error_kw=dict(lw=1, capsize=1, capthick=1),capsize=2, color=cmap(i*50), edgecolor='k')
    ax.set_xticks(x)
    ax.set_xticklabels(stats, fontsize=11)
    ax.legend(fontsize=11, ncol=3, loc='upper left')
    ax.grid(linestyle='--', axis='y')
    ax.set_axisbelow(True)

    a = np.stack((res_z2n['Burst_start_pos'], res_z2n['Burst_height'], res_z2n['Burst_freq'], \
    res_z2n['Burst_interarrival'], res_z2n['Burst_duration'], res_z2n['Burst_volume'])) # (3, 10)

    b = np.stack((res_knn['Burst_start_pos'], res_knn['Burst_height'], res_knn['Burst_freq'], \
        res_knn['Burst_interarrival'], res_knn['Burst_duration'], res_knn['Burst_volume'])) 
    c = np.stack((res_iter['Burst_start_pos'], res_iter['Burst_height'], res_iter['Burst_freq'], \
        res_iter['Burst_interarrival'], res_iter['Burst_duration'], res_iter['Burst_volume'])) 
    d = np.stack((res_plain['Burst_start_pos'], res_plain['Burst_height'], res_plain['Burst_freq'], \
        res_plain['Burst_interarrival'], res_plain['Burst_duration'], res_plain['Burst_volume'])) 
    e = np.stack((res_brits['Burst_start_pos'], res_brits['Burst_height'], res_brits['Burst_freq'], \
        res_brits['Burst_interarrival'], res_brits['Burst_duration'], res_brits['Burst_volume'])) 
    all_methods = [a,b,c,d,e]

    maxes = np.zeros(6)
    for i in range(len(all_methods)):
        for j in range(6):
            if max(all_methods[i][j]) > maxes[j]:
                maxes[j] = max(all_methods[i][j])
    means = np.zeros(6)
    for i in range(len(all_methods)):
        for j in range(6):
            if np.average(all_methods[i][j]) > means[j]:
                means[j] = np.average(all_methods[i][j])

    stats = ['Position', 'Height', 'Frequency', 'Interarrival\ndistance', 'Duration', \
         'Volume']
    methods = ['Zoom2Net', 'KNN', 'CoarseGrained', 'PlainTransformer', 'Brits']
    fig, ax = plt.subplots(1, 1, figsize=(6,3))
    cmap = plt.colormaps.get_cmap('Blues')
    diff = 0
    for i in range(5):
        r = all_methods[i].copy()
        for j in range(6):
    #         r[j] /= maxes[j]
            r[j] /= means[j] * 1.1
    #     x = np.arange(7)
    #     x = np.array([0,2,4,6,8,10,12])
        x = np.arange(0,18,3)
        width = 0.45
        mean = np.mean(r,axis=1)
        if i == 0:
            a = mean
        else:
            diff += sum(mean - a)/len(mean)
        std = np.std(r,axis=1)
        err_lo = mean - np.min(r,axis=1)
        err_hi = np.max(r,axis=1) - mean
        above_threshold = np.maximum(mean - 100, 0)
        below_threshold = np.minimum(mean, 100)
    #     print(x+(i-1.5)* width)
        print(methods[i], mean)

        ax.bar(x+(i-1.5)* width, below_threshold, width, label = methods[i],\
            capsize=2, color=cmap(i*50), edgecolor='k')
    ax.set_xticks(x+0.25)
    ax.set_xticklabels(stats, rotation=30)
    ax.legend(fontsize=10, ncol=3, loc='upper left')
    ax.grid(linestyle='--', axis='y')
    ax.set_axisbelow(True)