from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import joblib
import os
import random
import pickle

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.determinisdatic = True
    torch.backends.cudnn.benchmark = False
    print ("Seeded everything")
set_seed(0)

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class Network(torch.nn.Module):
  def __init__(self, input_dim, num_layers, num_units, output_dim):
    super(Network, self).__init__()
    
    layers = []
    layer_num_units = input_dim
    for _ in range(num_layers - 1):
        layers.append(torch.nn.Linear(layer_num_units, num_units))
        layers.append(torch.nn.ReLU())
        layer_num_units = num_units
    layers.append(torch.nn.Linear(layer_num_units, output_dim))
    layers.append(torch.nn.Sigmoid())

    self.classifier = torch.nn.Sequential(*layers)

  def forward(self, x):
    x = self.classifier(x)
    return x

def train_classifier(fiat_total, biat_total, fiat_mean, biat_mean, duration, fb_psec):
    with open("./datasets/vpn_data/len40_feat17.pickle", "rb") as fin:
        flat_res= pickle.load(fin)
        fin.close()

    X = []
    for i in range(len(flat_res)):
        a = flat_res[i][0]
        a = np.delete(a, [0,1,2,3,4,5,6,7,8,11,12,14]) 
        a = np.insert(a,0,duration[i])
        a = np.insert(a,1,fiat_total[i])
        a = np.insert(a,2,biat_total[i])
        a = np.insert(a,5,fiat_mean[i])
        a = np.insert(a,6,biat_mean[i])
        a = np.insert(a,8,fb_psec[i])
        X.append(a)
    Y = []
    for i in range(len(flat_res)):
        Y.append(flat_res[i][0][0])

    X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)
    X_train = np.log1p(X_train)
    X_test = np.log1p(X_test)
    print(np.isnan(X_train).any())
    print(np.isnan(X_test).any())
    print(np.isinf(X_train).any())
    print(np.isinf(X_test).any())
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    class Data(Dataset):
        def __init__(self, X_train, y_train):
            self.X = torch.from_numpy(np.array(X_train).astype(np.float32))
            self.y = torch.from_numpy(np.array(y_train).astype(np.float32))
            self.len = self.X.shape[0]
        
        def __getitem__(self, index):
            return self.X[index], self.y[index]
        def __len__(self):
            return self.len

    traindata = Data(X_train, Y_train)

    input_dim = X_train.shape[1]
    print("input_dim: ", input_dim)
    num_layers = 5
    num_units = 100
    output_dim = 1

    batch_size = 1000
    trainloader = DataLoader(traindata, batch_size=batch_size, 
                            shuffle=True, num_workers=0)
    testtrainloader = DataLoader(traindata, batch_size=batch_size, 
                            shuffle=True, num_workers=0)

    testdata = Data(X_test, Y_test)
    testloader = DataLoader(testdata, batch_size=batch_size, 
                            shuffle=True, num_workers=0)

    
    for _ in range(1):
        random.seed(0)
        set_seed(0)
        mlp_classifier = Network(input_dim, num_layers, num_units, output_dim)
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.SGD(mlp_classifier.parameters(), lr=0.1)
        epochs = 10000
        epoch_list = []
        training_aucs = []
        training_loss = []
        testing_loss = []
        testing_aucs = []
        
        training_accs = []
        testing_accs = []
        early_stopper = EarlyStopper(patience=20, min_delta=0.00001)
        for epoch in range(epochs):
            mlp_classifier.train()
            losses = 0
            cnt = 0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                labels = labels.view(-1, 1)
                optimizer.zero_grad()
                outputs = mlp_classifier(inputs)
                # print(labels)
                # print(outputs)
                loss = criterion(outputs, labels)
                losses += loss.detach().cpu()
                cnt += 1
                loss.backward()
                optimizer.step()
            training_loss.append(losses / cnt)
        mlp_classifier.eval()
        prediction = []
        ground_truth = []
        for i, testdata in enumerate(testloader):
            inputs, labels = testdata
            pred = mlp_classifier(inputs)
            prediction.append(torch.round(torch.squeeze(pred)))
            ground_truth.append(labels)
        prediction_results = torch.cat(prediction).detach().numpy()
        ground_truth_results = torch.cat(ground_truth).detach().numpy()
        
        auc_score = roc_auc_score(ground_truth_results, prediction_results)
        accuracy = accuracy_score(ground_truth_results, prediction_results)
        print(f"Accuracy: {accuracy}, AUC score: {auc_score}")
        print("------------")