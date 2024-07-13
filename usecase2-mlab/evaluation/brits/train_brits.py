import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import itertools

from evaluation.brits.model import BritsModel
import evaluation.brits.utils as utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

FEATURE_NUM = 8
def parse_delta(masks, dir_, WINDOW_SIZE):
    deltas = []
    # if masks.shape[0] != 8:
    #     print('not')
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

def convert_odd(data):
    a = np.zeros((8,6,300))
    for i in range(8):
        for j in range(6):
            a[i,0,j*50] = data[i*2+1][3][j*50]
            a[i,1,(j+1)*50-1] = data[i*2+1][0][j*50] * 2
            a[i,2,j*50] = data[i*2][3][j*50] 
            a[i,3,(j+1)*50-1] = data[i*2][0][j*50] 
            a[i,4,(j+1)*50-1] = data[i*2][1][j*50]
            a[i,5,(j+1)*50-1] = data[i*2][2][j*50]
    return a

def prepare_brits_data(config,train_dataset, test_dataset):
    WINDOW_SKIP = config.window_skip
    WINDOW_SIZE = config.window_size
    COARSE = config.zoom_in_factor
    all_rec_train = []
    for i in range(len(train_dataset)):
        periodic = np.zeros((1,WINDOW_SIZE))
        time = [int(a) for a in train_dataset[i][2]] #int(train_dataset[i][2])
        periodic[0,time] = train_dataset[i][1][time]
        value = np.transpose(np.concatenate((periodic, train_dataset[i][0])))
        masks = np.ones(value.shape) # 300, 8
        for k in range(WINDOW_SIZE):
            if k not in time:
                masks[k, 0] = 0
        evals = value.copy()
        evals[:,0] = train_dataset[i][1]
        eval_masks = np.zeros(evals.shape)
        eval_masks[:,0] = 1
        rec = {}
        rec['forward'] = parse_rec(value, masks, evals, eval_masks, dir_='forward', WINDOW_SIZE=WINDOW_SIZE)
        rec['backward'] = parse_rec(value[::-1], masks[::-1], evals[::-1], eval_masks[::-1], dir_='backward', WINDOW_SIZE=WINDOW_SIZE)
        all_rec_train.append(rec)
        
    all_rec_test = []
    for i in range(len(test_dataset)):
        periodic = np.zeros((1,WINDOW_SIZE))
        # time = int(test_dataset[i][2])
        time = [int(a) for a in test_dataset[i][2]]
        periodic[0,time] = test_dataset[i][1][time]
        value = np.transpose(np.concatenate((periodic, test_dataset[i][0])))
        masks = np.ones(value.shape) # 300, 8
        for k in range(WINDOW_SIZE):
            if k not in time:
                masks[k, 0] = 0
        evals = value.copy()
        evals[:,0] = test_dataset[i][1]
        eval_masks = np.zeros(evals.shape)
        eval_masks[:,0] = 1
        rec = {}
        rec['forward'] = parse_rec(value, masks, evals, eval_masks, dir_='forward', WINDOW_SIZE=WINDOW_SIZE)
        rec['backward'] = parse_rec(value[::-1], masks[::-1], evals[::-1], eval_masks[::-1], dir_='backward', WINDOW_SIZE=WINDOW_SIZE)
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

def run_inference(config, test_dataset, model, data_3s_train, \
                data_3s_test, data_3to6s_train, data_3to6s_test, data_6to9s_train, data_6to9s_test, \
                index_of_3s, index_of_3to6s, index_of_6to9s):
    WINDOW_SKIP = config.window_skip
    WINDOW_SIZE = config.window_size
    COARSE = config.zoom_in_factor
    model.eval()

    pred9s_brits = []
    true9s_brits = []
    pred6s_brits = []
    true6s_brits = []
    pred3s_brits = []
    true3s_brits = []
    for i in range(data_3s_test):
        rec = convert_brits(test_dataset[i], WINDOW_SIZE, COARSE)
        iter_rec = get_loader(rec, batch_size=1, shuffle = False)
        d = next(iter(iter_rec))
        ret = model.run_on_batch(utils.to_var(d), optimizer=None)
        eval_masks = ret['eval_masks'].data.cpu().numpy()
        imputation = ret['imputations'].data.cpu().numpy()
        pred = imputation[np.where(eval_masks == 1)].tolist()

        s3 = pred
        ground_thruth_3s = test_dataset[i][1]
        a = index_of_3s[data_3s_train+i]
        b = np.where(index_of_3to6s == a)[0]
        c = np.where(index_of_6to9s == a)[0]
        exist_3to6 = False
        exist_6to9 = False
        if len(b) != 0 and b[0] > data_3to6s_train:
            exist_3to6 = True
            test_index = b[0] - data_3to6s_train + data_3s_test

            rec = convert_brits(test_dataset[test_index], WINDOW_SIZE, COARSE)
            iter_rec = get_loader(rec, batch_size=1, shuffle = False)
            d = next(iter(iter_rec))
            ret = model.run_on_batch(utils.to_var(d), optimizer=None)
            eval_masks = ret['eval_masks'].data.cpu().numpy()
            imputation = ret['imputations'].data.cpu().numpy()
            pred = imputation[np.where(eval_masks == 1)].tolist()

            s3to6 = pred
            ground_thruth_3to6s = test_dataset[test_index][1]
        if exist_3to6 == True and len(c) != 0 and c[0] > data_6to9s_train:
            exist_6to9 = True
            test_index2 = c[0] - data_6to9s_train + data_3s_test + data_3to6s_test

            rec = convert_brits(test_dataset[test_index2], WINDOW_SIZE, COARSE)
            iter_rec = get_loader(rec, batch_size=1, shuffle = False)
            d = next(iter(iter_rec))
            ret = model.run_on_batch(utils.to_var(d), optimizer=None)
            eval_masks = ret['eval_masks'].data.cpu().numpy()
            imputation = ret['imputations'].data.cpu().numpy()
            pred = imputation[np.where(eval_masks == 1)].tolist()
            
            s6to9 = pred
            ground_thruth_6to9s = test_dataset[test_index2][1]
        if exist_3to6 == True and exist_6to9 == True:
            pred9s_brits.append(np.concatenate((s3, s3to6, s6to9)))
            true9s_brits.append(np.concatenate((ground_thruth_3s, ground_thruth_3to6s, ground_thruth_6to9s)))
        elif exist_3to6 == True and exist_6to9 == False:
            pred6s_brits.append(np.concatenate((s3, s3to6)))
            true6s_brits.append(np.concatenate((ground_thruth_3s, ground_thruth_3to6s)))
        elif exist_3to6 == False and exist_6to9 == False:
            pred3s_brits.append(s3)
            true3s_brits.append(ground_thruth_3s)
    pred9s_brits = np.array(pred9s_brits)
    true9s_brits = np.array(true9s_brits)
    pred6s_brits = np.array(pred6s_brits)
    true6s_brits = np.array(true6s_brits)
    pred3s_brits = np.array(pred3s_brits)
    true3s_brits = np.array(true3s_brits)
    return [true9s_brits, true6s_brits, true3s_brits], [pred9s_brits, pred6s_brits, pred3s_brits]

def convert_brits(data, WINDOW_SIZE, COARSE):
    periodic = np.zeros((1,WINDOW_SIZE))
    time = [int(a) for a in data[2]]
    periodic[0,time] = data[1][time]
    value = np.transpose(np.concatenate((periodic, data[0])))
    masks = np.ones(value.shape) # 300, 8
    for k in range(WINDOW_SIZE):
        if k not in time:
            masks[k, 0] = 0
    evals = value.copy()
    evals[:,0] = data[1]
    eval_masks = np.zeros(evals.shape)
    eval_masks[:,0] = 1
    rec = {}
    rec['forward'] = parse_rec(value, masks, evals, eval_masks, dir_='forward', WINDOW_SIZE=WINDOW_SIZE)
    rec['backward'] = parse_rec(value[::-1], masks[::-1], evals[::-1], eval_masks[::-1], dir_='backward', WINDOW_SIZE=WINDOW_SIZE)
    return [rec]