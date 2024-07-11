import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

from evaluation.brits.model import BritsModel
import evaluation.brits.utils as utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

FEATURE_NUM = 9
def parse_delta(masks, dir_, WINDOW_SIZE):
    deltas = []

    for h in range(WINDOW_SIZE):
        if h == 0:
            deltas.append(np.ones(FEATURE_NUM))
        else:
            deltas.append(np.ones(FEATURE_NUM) + (1 - masks[h]) * deltas[-1])

    return np.array(deltas)


def parse_rec(value, masks, evals, eval_masks, dir_, WINDOW_SIZE):
    deltas = parse_delta(masks, dir_, WINDOW_SIZE)
    rec = {}
    rec['values'] = np.nan_to_num(value).tolist()
    rec['masks'] = masks.astype('int32').tolist()
    # imputation ground-truth
    rec['evals'] = np.nan_to_num(evals).tolist()
    rec['eval_masks'] = eval_masks.astype('int32').tolist()
    rec['deltas'] = deltas.tolist()

    return rec

def prepare_brits_data(config,train_dataset, test_dataset):
    WINDOW_SKIP = config.window_skip
    WINDOW_SIZE = config.window_size
    COARSE = config.zoom_in_factor
    indexes = []
    for i in range(92):
        a = list(range(i))
        b = list(range(i+1,92))
        indexes.append((a+b))
    all_rec_train = []
    for i in range(len(train_dataset)):
        t = train_dataset[i][0][:,[0,2,4,5],:]
        t[:,0,:] = t[:,0,:]/3
        t[:,3,:] = t[:,3,:]/6
        for j in range(92):
            r = np.zeros((9,WINDOW_SIZE))
            b = np.sum(t[indexes[j]], axis=0)/91/2
            for z in range(WINDOW_SIZE//COARSE):
                r[0][z*COARSE] = train_dataset[i][1][j][z*COARSE]
                r[1][(z+1)*COARSE-1] = t[j,0, z]
                r[2][(z+1)*COARSE-1] = t[j,1, z]
                r[3][(z+1)*COARSE-1] = t[j,2, z]
                r[4][(z+1)*COARSE-1] = t[j,3, z]
                r[5][(z+1)*COARSE-1] = b[0,z]
                r[6][(z+1)*COARSE-1] = b[1,z]
                r[7][(z+1)*COARSE-1] = b[2,z]
                r[8][(z+1)*COARSE-1] = b[3,z]
            rec = {}
            value = np.transpose(r)
            masks = np.ones(value.shape) # 300, 12
            for k in range(WINDOW_SIZE//COARSE):
                masks[k*COARSE+1:(k+1)*COARSE, 0] = 0
            evals = value.copy()
            evals[:,0] = train_dataset[i][1][j]
            eval_masks = np.zeros(evals.shape)
            eval_masks[:,0] = 1
            rec['forward'] = parse_rec(value, masks, evals, eval_masks, dir_='forward', WINDOW_SIZE=1000)
            rec['backward'] = parse_rec(value[::-1], masks[::-1], evals[::-1], eval_masks[::-1], \
                                dir_='backward', WINDOW_SIZE=1000)
            all_rec_train.append(rec)

    all_rec_test = []
    for i in range(len(test_dataset)):
        t = test_dataset[i][0][:,[0,2,4,5],:]
        t[:,0,:] = t[:,0,:]/3
        t[:,3,:] = t[:,3,:]/6
        for j in range(92):
            r = np.zeros((9,WINDOW_SIZE))
            b = np.sum(t[indexes[j]], axis=0)/91/2
            for z in range(WINDOW_SIZE//COARSE):
                r[0][z*COARSE] = test_dataset[i][1][j][z*COARSE]
                r[1][(z+1)*COARSE-1] = t[j,0, z]
                r[2][(z+1)*COARSE-1] = t[j,1, z]
                r[3][(z+1)*COARSE-1] = t[j,2, z]
                r[4][(z+1)*COARSE-1] = t[j,3, z]
                r[5][(z+1)*COARSE-1] = b[0,z]
                r[6][(z+1)*COARSE-1] = b[1,z]
                r[7][(z+1)*COARSE-1] = b[2,z]
                r[8][(z+1)*COARSE-1] = b[3,z]

            rec = {}
            value = np.transpose(r)
            masks = np.ones(value.shape) # 300, 12
            for k in range(WINDOW_SIZE//COARSE):
                masks[k*COARSE+1:(k+1)*COARSE, 0] = 0
            evals = value.copy()
            evals[:,0] = test_dataset[i][1][j]
            eval_masks = np.zeros(evals.shape)
            eval_masks[:,0] = 1
            rec['forward'] = parse_rec(value, masks, evals, eval_masks, dir_='forward', WINDOW_SIZE=1000)
            rec['backward'] = parse_rec(value[::-1], masks[::-1], evals[::-1], eval_masks[::-1], dir_='backward', WINDOW_SIZE=1000)
            all_rec_test.append(rec)    
    data_iter_train = get_loader(all_rec_train, batch_size=64)
    data_iter_test = get_loader(all_rec_test, batch_size=16)
    return data_iter_train, data_iter_test

class MySet(Dataset):
    def __init__(self, all_rec):
        super(MySet, self).__init__()
        self.content = all_rec

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        rec = self.content[idx]
        return rec

def collate_fn(recs):
    forward = list(map(lambda x: x['forward'], recs))
    backward = list(map(lambda x: x['backward'], recs))

    def to_tensor_dict(recs):
        values = torch.FloatTensor(list(map(lambda r: r['values'], recs)))
        masks = torch.FloatTensor(list(map(lambda r: r['masks'], recs)))
        deltas = torch.FloatTensor(list(map(lambda r: r['deltas'], recs)))

        evals = torch.FloatTensor(list(map(lambda r: r['evals'], recs)))
        eval_masks = torch.FloatTensor(list(map(lambda r: r['eval_masks'], recs)))

        return {'values': values, 'masks': masks, 'deltas': deltas, 'evals': evals, 'eval_masks': eval_masks}

    ret_dict = {'forward': to_tensor_dict(forward), 'backward': to_tensor_dict(backward)}

    return ret_dict

def get_loader(all_rec, batch_size = 64, shuffle = True):
    data_set = MySet(all_rec)
    data_iter = DataLoader(dataset = data_set, \
                              batch_size = batch_size, \
                              num_workers = 4, \
                              shuffle = shuffle, \
                              pin_memory = True, \
                              collate_fn = collate_fn)

    return data_iter

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
    #         print(f"early: {validation_loss}")
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def train(model, data_iter_train, data_iter_test):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    early_stopper = EarlyStopper(patience=5, min_delta=0.000001)
    for epoch in range(100):
        model.train()
        run_loss = 0.0
        for idx, data in enumerate(data_iter_train):
            data = utils.to_var(data)
            ret = model.run_on_batch(data, optimizer, epoch)

            run_loss += ret['loss'].item()
            if idx % 20 == 0:
                print(f"Epoch: {epoch}, {(idx + 1) * 100.0 / len(data_iter_train)}, loss: {run_loss / (idx + 1.0)}")
        mae, mre = evaluate(model, data_iter_test)
        break ####TODO!!!!!!!!!!!! Remove
        if early_stopper.early_stop(mae):             
            break

def train_brits(data_iter_train, data_iter_test, WINDOW_SIZE):
    selected_model = 'brits'
    selected_hid_size = 64
    selected_impute_weight = 1
    selected_label_weight = 0
    model = BritsModel(selected_hid_size, selected_impute_weight, selected_label_weight, WINDOW_SIZE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total params is {}'.format(total_params))

    if torch.cuda.is_available():
        model = model.cuda()
    train(model, data_iter_train, data_iter_test)

    return model

def evaluate(model, val_iter):
    model.eval()

    labels = []
    preds = []

    evals = []
    imputations = []

    save_impute = []
    save_label = []

    for idx, data in enumerate(val_iter):
        data = utils.to_var(data)
        ret = model.run_on_batch(data, None)
        eval_masks = ret['eval_masks'].data.cpu().numpy()
        eval_ = ret['evals'].data.cpu().numpy()
        imputation = ret['imputations'].data.cpu().numpy()

        evals += eval_[np.where(eval_masks == 1)].tolist()
        imputations += imputation[np.where(eval_masks == 1)].tolist()
        
    evals = np.asarray(evals)
    imputations = np.asarray(imputations)
    mae = np.abs(evals - imputations).mean()
    mre = np.abs(evals - imputations).sum() / np.abs(evals).sum()
    print('MAE: ', np.abs(evals - imputations).mean())

    print('MRE: ', np.abs(evals - imputations).sum() / np.abs(evals).sum())
    return mae, mre

def run_inference(config, test_dataset, model, rackdata_len):
    WINDOW_SKIP = config.window_skip
    WINDOW_SIZE = config.window_size
    COARSE = config.zoom_in_factor
    model.eval()
    indexes = []
    for i in range(92):
        a = list(range(i))
        b = list(range(i+1,92))
        indexes.append((a+b))
    num_intervals = len(np.arange(0,2000,WINDOW_SKIP))-1
    num_WINDOW = len(np.arange(0,2000,WINDOW_SIZE))
    skipped = WINDOW_SIZE // WINDOW_SKIP
    res_true_brits = np.zeros((rackdata_len, 92, num_WINDOW, WINDOW_SIZE))
    res_pred_brits = np.zeros((rackdata_len, 92, num_WINDOW, WINDOW_SIZE))
    for q in range(92):
        for i in range(rackdata_len):
            cnt = 0
            for j in range(i*num_intervals, (i+1)*num_intervals):
                if (j < num_intervals and j % skipped == 0) or \
                (j >= num_intervals * i and (j - num_intervals * i) % skipped == 0):
                    t = test_dataset[j][0][:,[0,2,4,5],:]
                    t[:,0,:] = t[:,0,:]/3
                    t[:,3,:] = t[:,3,:]/6
                    r = np.zeros((9,WINDOW_SIZE))
                    b = np.sum(t[indexes[q]], axis=0)/91/2
                    for z in range(WINDOW_SIZE//COARSE):
                        r[0][z*COARSE] = test_dataset[j][1][q][z*COARSE]
                        r[1][(z+1)*COARSE-1] = t[q,0, z]
                        r[2][(z+1)*COARSE-1] = t[q,1, z]
                        r[3][(z+1)*COARSE-1] = t[q,2, z]
                        r[4][(z+1)*COARSE-1] = t[q,3, z]
                        r[5][(z+1)*COARSE-1] = b[0,z]
                        r[6][(z+1)*COARSE-1] = b[1,z]
                        r[7][(z+1)*COARSE-1] = b[2,z]
                        r[8][(z+1)*COARSE-1] = b[3,z]
                    rec = convert_brits([r, test_dataset[j][1][q]], WINDOW_SIZE, COARSE)
                    iter_rec = get_loader(rec, batch_size=1, shuffle = False)
                    d = next(iter(iter_rec))
                    ret = model.run_on_batch(utils.to_var(d), optimizer=None)
                    eval_masks = ret['eval_masks'].data.cpu().numpy()
                    imputation = ret['imputations'].data.cpu().numpy()
                    pred = imputation[np.where(eval_masks == 1)].tolist()
                    res_true_brits[i,q,cnt,:] = test_dataset[j][1][q]
                    res_pred_brits[i,q,cnt,:] = pred
                    cnt += 1
                    # print(cnt)
                    
    res_true_brits = np.reshape(res_true_brits, (rackdata_len,92,num_WINDOW*WINDOW_SIZE))
    res_pred_brits = np.reshape(res_pred_brits, (rackdata_len,92,num_WINDOW*WINDOW_SIZE))
    res_true_brits = np.reshape(res_true_brits, (rackdata_len*92,num_WINDOW*WINDOW_SIZE))
    res_pred_brits = np.reshape(res_pred_brits, (rackdata_len*92,num_WINDOW*WINDOW_SIZE))

    return res_true_brits, res_pred_brits

def convert_brits(data, WINDOW_SIZE, COARSE):
    rec = {}
    value = np.transpose(data[0])
    masks = np.ones(value.shape) # 300, 12
    for k in range(WINDOW_SIZE//COARSE):
        masks[k*COARSE+1:(k+1)*COARSE, 0] = 0
    evals = value.copy()
    evals[:,0] = data[1]
    eval_masks = np.zeros(evals.shape)
    eval_masks[:,0] = 1
    rec['forward'] = parse_rec(value, masks, evals, eval_masks, dir_='forward', WINDOW_SIZE=1000)
    rec['backward'] = parse_rec(value[::-1], masks[::-1], evals[::-1], eval_masks[::-1], dir_='backward', WINDOW_SIZE=1000)
    return [rec]