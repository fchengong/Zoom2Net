import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader

from model_training.transformer import TSTransformerEncoder
from model_training import utils

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.determinisdatic = True
    torch.backends.cudnn.benchmark = False
    print ("Seeded everything")

def train_z2n(config, train_dataset_processed, test_dataset_processed, seed):
    set_seed(seed)
    WINDOW_SIZE = config.window_size
    WINDOW_SKIP = config.window_skip
    COARSE = config.zoom_in_factor
    feat_dim = config.feat_dim

    # Training setup
    dim_output = 1
    d_model = config.d_model
    n_heads = config.n_heads
    num_layers = config.num_layers
    max_len = 1
    dim_feedforward = config.dim_feedforward
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Initialize model")
    model = TSTransformerEncoder(
        window_size=WINDOW_SIZE,
        feat_dim=feat_dim,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        max_len=max_len,
        dim_feedforward=dim_feedforward,
        dim_output=dim_output,
        dropout=config.dropout,
        activation='relu',
        norm='LayerNorm'
        )
    
    optimizer = optim.AdamW(model.parameters(), lr = config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    batch_size = config.batch_size
    train_loader = DataLoader(train_dataset_processed, batch_size=batch_size, num_workers=4, shuffle=True)
    test_loader = DataLoader(test_dataset_processed, batch_size=1, num_workers=1, shuffle=False)
    mu = config.mu_lagrange
    update_train_dataset = train_dataset_processed
    total_constr_error = []
    sum_constr_error = []
    retrans_constr_error = []
    prev_constraint_loss = np.inf
    constrain_loss_decrease = 0
    if(device != torch.device("cpu")):
        model.cuda()
    iteration = 0
    training_loss = []
    val_loss = []
    test_violation = []
    
    print("Start training")
    while True:
        # Continue until test data constraint violation stops decreasing
        print(f"Largrange iteration {iteration}")
        epoch = 0
        train_early_stopper = utils.EarlyStopper(patience=5, min_delta=0.00000001)
        test_early_stopper = utils.EarlyStopper(patience=1, min_delta=10000)
        while True:
            # Continue until test data loss stops decreasing
            print(f"Epoch {epoch}")
            scheduler, train_loss, validation_loss = utils.train_epoch(model, train_loader, \
            test_loader, optimizer, scheduler, config, mu=mu)
            training_loss.append(train_loss) 
            val_loss.append(validation_loss.detach().cpu())
            if train_early_stopper.early_stop(validation_loss):             
                break
            epoch += 1
        train_loader = DataLoader(update_train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
        check_constr, max_constr, \
                update_train_dataset = utils.check_constraints(model, train_loader, mu=mu, \
                batch_size=batch_size, WINDOW_SIZE=WINDOW_SIZE, COARSE=COARSE)
        check_constr = check_constr.detach().cpu()
        print(f"Epoch {epoch} finish, training data constraint loss {check_constr}")
        total_constr_error.append(check_constr)
        test_constr_violation = utils.test_constraint(model, test_dataset_processed, \
                WINDOW_SIZE=WINDOW_SIZE, COARSE=COARSE, device=device)
        test_violation.append(test_constr_violation)
        print(f"Iteration {iteration} finish, tesing data constraint loss {test_constr_violation}")
        if test_early_stopper.early_stop(test_constr_violation):             
                break
        train_loader = DataLoader((update_train_dataset), \
                                batch_size=batch_size, num_workers=4, shuffle=True)
        mu = mu * 1.5
        iteration += 1

    torch.save(model.state_dict(), config.save_model_dir)