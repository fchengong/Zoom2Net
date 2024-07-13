import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from scipy.stats import ttest_ind
from scipy import stats
import ot
import pickle

def percen(x, y, number):
    p_true = np.percentile(x, number)
    p_pred = np.percentile(y, number)
    if p_true == 0:
        return (abs(p_true - p_pred) / 1)
    else:
        return (abs(p_true - p_pred) / p_true)

def error(pred, true, web_bytes):
    pred_sum = np.cumsum(pred)
    true_sum = np.cumsum(true)
    pred_time = np.where(pred_sum >= web_bytes)[0]
    true_time = np.where(true_sum >= web_bytes)[0]
    if len(pred_time) != 0 and len(true_time) != 0:
        if true_time[0] == 0:
            return abs(pred_time[0] - true_time[0])
        else:
            return abs(pred_time[0] - true_time[0]) / true_time[0]
    else:
        return None

def myFunc(e):
    return e['rank']
def s(a):
    return a[1]

def downstream_task(res_true, res_pred):
    mse = []
    acorr = []
    emd = []
    p99 = []
    cdf_dist = []
    cdf_error = []
    with open("./datasets/mlab_data/top_websites.pickle", "rb") as fin:
        top_websites= pickle.load(fin)
        fin.close()
    with open("./datasets/mlab_data/site_bytes2.pickle", "rb") as fin:
        site_bytes= pickle.load(fin)
        fin.close()
    website_rank_bytes = []
    for i in site_bytes.keys():
        if i not in top_websites:
            continue
        website_rank_bytes.append({'website': i, 'rank': top_websites.index(i), 'bytes': site_bytes[i]})
    website_rank_bytes.sort(key=myFunc)

    a = []
    for i in website_rank_bytes[0:10]:
        a.append([i['website'], i['bytes']])
    
    sorted_web = sorted(a, key=s)

    for j in range(len(res_true)):
        data_true = res_true[j]
        data_pred = res_pred[j]
        for i in range(len(data_true)):
            x = data_true[i]
            ground_thruth = data_pred[i]
            mse.append(mean_squared_error(x, ground_thruth))
            a1 = autocorr(x)
            a2 = autocorr(ground_thruth)
            acorr.append(mean_squared_error(a1,a2))
            p99.append(percen(x, ground_thruth, 99))
            emd.append(ot.wasserstein_1d(x, ground_thruth))
            cdf_error.append(mean_squared_error(np.cumsum(x), np.cumsum(ground_thruth)))
    mse_final = sum(mse)/len(mse)
    acorr_final = sum(acorr)/len(acorr)
    p99_final = sum(p99)/len(p99)
    emd_final = sum(emd)/len(emd)
    cdf_error_final = sum(cdf_error)/len(cdf_error)

    result = {}
    result['MSE'] = (mse)
    result['emd'] = (emd)
    result['Autocorrelation'] = (acorr)
    result['99_percentile'] = (p99)

    print('mse_final', mse_final)
    print('acorr_final', acorr_final)
    print('p99_final', p99_final)
    print('emd_final', emd_final)
    print('cdf_error_final', cdf_error_final)

    error_model = []
    skip = ['youtube-mp3.org', 'seasonvar.ru', 'gfycat.com']
    normalization_bytes = 594940
    for j in sorted_web:
        # web_name = j['website']
        web_name = j[0]
        if web_name in skip:
            continue
        a = []
        # web_bytes = j['bytes']
        web_bytes = j[1]
        for k in range(len(res_true)):
            data_true = res_true[k]
            data_pred = res_pred[k]
            for i in range(len(data_true)):
                pred = data_pred[i] * normalization_bytes
                true = data_true[i] * normalization_bytes
                e = error(pred, true,web_bytes)
                if e != None:
                    a.append(e)
        result[web_name] = sum(a) / len(a)
        print(web_name+": ", sum(a) / len(a))
    return result

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[int(result.size/2):]

def burst_detection(data, th, low_threshold, high_threshold):
    res = []
    bursts_val = []
    bursts_index = []
    height = []
    inc_rate = []
    dec_rate = []
    peak_pos = []
    burst = False
    for i in range(len(data)):
        temp = []
        temp_ind = []
        if burst == False and data[i] > low_threshold:
            temp.append(data[i])
            temp_ind.append(i)
            j = i 
            while j + 1 < len(data) -1:
                j = j + 1
                if data[j] > low_threshold:
                    temp.append(data[j])
                    temp_ind.append(j)
                else:
                    dec = j
                    dec_v = data[j]
                    break
            if len(temp) > 1:
                ind = temp.index(max(temp))
                if (ind > 0 and (temp[ind] - temp[0])/ind > th and temp[ind] > high_threshold) \
                    or (ind == 0 and temp[ind] > high_threshold):
                    burst = True
                    bursts_val.append(temp)
                    bursts_index.append(temp_ind)
            elif len(temp) == 1 and temp[0] > high_threshold:
                bursts_val.append(temp)
                bursts_index.append(temp_ind)
        elif burst == True and data[i] < low_threshold:
            burst = False
    return bursts_val, bursts_index