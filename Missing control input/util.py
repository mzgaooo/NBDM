import torch
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from data_loader import Dataset_PEMS, Dataset_PM25, Dataset_Temp


data_dict = {
    'PEMS04': Dataset_PEMS,
    'PEMS08': Dataset_PEMS,
    'PEMSBAY': Dataset_PEMS,
    'PM25': Dataset_PM25,
    'Temp': Dataset_Temp,
}

def data_provider(args, flag):
    Data = data_dict[args.data]

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.input_len, args.pred_len],
        features=args.features,
        target=args.target,
    )

    print(flag, len(data_set))

    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True)
    return data_set, data_loader


def adjust_learning_rate(optimizer, epoch, args, scheduler=None, printout=True):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.lr * (0.5 ** (epoch // 20))}
    elif args.lradj == 'type2':
        lr_adjust = {epoch: args.lr * (0.5 ** ((epoch-1) // 10))}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def DMD_Learner(args):
    if args.data in ['PEMS04','PEMS08', 'PEMSBAY']:
        data = np.load(os.path.join(args.root_path, args.data_path))['data'] #T*N*C
        train_data = data[:int(0.6 * len(data)), :, 0] #T*N
    elif args.data == 'PM25':
        csv_file = os.path.join(args.root_path, args.data_path)
        df = pd.read_csv(csv_file)
        pm = wide(df, 'PM2.5')
        pm_data = clean_smooth(pm) 
        lat_data = wide(df, 'Latitude')
        lon_data = wide(df, 'Longitude')
        data = np.stack([pm_data, lat_data, lon_data], axis=-1)   # [T, N, 3]
         
        train_data = data[:int(0.6 * len(data)), :, 0] #T*N
    elif args.data == 'Temp':
        data = np.loadtxt(os.path.join(args.root_path, args.data_path))
        train_data = data[:int(0.6 * len(data)), :] #T*N

    scaler = StandardScaler()
    scaler.fit(train_data)
    train_scaled = scaler.transform(train_data)

    if not os.path.exists(f'{args.data}_Dynamic_graph.npy'):
        X = train_scaled[:-1, :]       
        Y = train_scaled[1:, :]
        A_edmd  = np.linalg.lstsq(X, Y, rcond=None)[0] #N*N 
        A_edmd  = np.abs(A_edmd)
        np.save(f'{args.data}_Dynamic_graph.npy', A_edmd)
        
    A_edmd = np.load(f'{args.data}_Dynamic_graph.npy')

    return A_edmd


def wide(df, col):
    tmp = df[['Station code', 'Measurement date', col]].copy()
    tmp['id'] = tmp['Station code'].astype(str)
    tmp = tmp.set_index(['Measurement date', 'id'])[col].unstack()
    return tmp.values[:17520, :]   # [T, N]

def clean_smooth(data):
    T, N = data.shape
    for t in range(T-1):
        for n in range(N):
            if abs(data[t, n] - data[t+1, n]) >= 50:
                data[t+1, n] = data[t, n]
    noise = np.random.randint(2, 4, size=(T, N))
    data = data + noise
    data = savgol_filter(data, 53, 3, axis=0)
    return data