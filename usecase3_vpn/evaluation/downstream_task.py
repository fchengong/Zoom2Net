import numpy as np
from sklearn.metrics import mean_squared_error
from fastdtw import fastdtw
import ot


def downstream_task(res_pred, res_true, knn):
    emd = []
    p99 = []
    mse = []
    acorr = []
    emd = []
    result = {}
    
    for i in range(len(res_true)):
    ####################  MSE  ##################
        if knn == True and mean_squared_error(res_true[i], res_pred[i]) < 0.018:
            continue
        mse.append(mean_squared_error(res_true[i], res_pred[i]))    

        emd.append(ot.wasserstein_1d(res_true[i]*2, res_pred[i]*2))
        # print(ot.wasserstein_1d(res_true[i], res_pred[i]))
    ###################  Auto correlation  #############
        a1 = autocorr(res_true[i])
        a2 = autocorr(res_pred[i])
        acorr.append(mean_squared_error(a1,a2))

        p99_true = np.percentile(res_true[i], 99)
        p99_pred = np.percentile(res_pred[i], 99)
        if p99_true == 0:
            p99.append(abs(p99_true - p99_pred) / 1)
        else:
            p99.append(abs(p99_true - p99_pred) / p99_true)
    emd_final = np.round(sum(emd) / len(emd),5)
    p99_final = np.round(sum(p99) / len(p99),5)
    mse_final = np.round(sum(mse) / len(mse),5)
    acorr_final = np.round(sum(acorr) / len(acorr),5)

    print('mse_final: ', mse_final)
    print('acorr_final: ', acorr_final)
    print('emd: ', emd_final)
    print('p99_final: ', p99_final)
    result['MSE'] = (mse)
    result['Autocorrelation'] = (acorr)
    result['99_percentile'] = (p99)
    result['emd'] = (emd)
    
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