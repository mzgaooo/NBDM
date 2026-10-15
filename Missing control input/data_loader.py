import os
import warnings
import numpy as np
import torch
import pandas as pd
from scipy.signal import savgol_filter
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


class Dataset_PM25(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='M',
                 data_path='Measurement_summary.csv', target='OT', scale=True):
        self.input_len = size[0]
        self.pred_len = size[1]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        csv_file = os.path.join(self.root_path, self.data_path)
        df = pd.read_csv(csv_file)

        # 构造 stations×time 宽表（三通道）
        pm = self._wide(df, 'PM2.5')
        pm_data = self._clean_smooth(pm) 
        lat_data = self._wide(df, 'Latitude')
        lon_data = self._wide(df, 'Longitude')
        data = np.stack([pm_data, lat_data, lon_data], axis=-1)   # [T, N, 3]

        train_ratio, val_ratio = 0.6, 0.2
        n = len(data)
        train_data = data[:int(n * train_ratio)]
        val_data = data[int(n * train_ratio): int(n * (train_ratio + val_ratio))]
        test_data = data[int(n * (train_ratio + val_ratio)):]

        total_data = [train_data, val_data, test_data]
        cur_data = total_data[self.set_type]       # [T', N, 3]

        if self.scale:
            self.scaler.fit(train_data[:, :, 0])
            cur_data[:, :, 0] = self.scaler.transform(cur_data[:, :, 0])  

        self.data_x = cur_data
        self.data_y = cur_data

    def _wide(self, df, col):
        tmp = df[['Station code', 'Measurement date', col]].copy()
        tmp['id'] = tmp['Station code'].astype(str)
        tmp = tmp.set_index(['Measurement date', 'id'])[col].unstack()
        # 取前 17520 列,不然会nan
        return tmp.values[:17520, :]   # [T, N]

    def _clean_smooth(self, data):
        # data: [T, N]
        T, N = data.shape
        for t in range(T-1):
            for n in range(N):
                if abs(data[t, n] - data[t+1, n]) >= 50:
                    data[t+1, n] = data[t, n]
        noise = np.random.randint(2, 4, size=(T, N))
        data = data + noise
        data = savgol_filter(data, 53, 3, axis=0)
        return data

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.input_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end, :, 0]
        seq_y = self.data_y[r_begin:r_end, :, 0:1]

        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.input_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Temp(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='M',
                 data_path='eighteen.txt', target='OT', scale=True):
        self.input_len  = size[0]
        self.pred_len   = size[1]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features   = features
        self.target     = target
        self.scale      = scale
        self.root_path  = root_path
        self.data_path  = data_path

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        raw = np.loadtxt(os.path.join(self.root_path, self.data_path))
        data = raw[:, :, np.newaxis]  # [T, N] -> [T, N, 1]

        n = len(data)
        train_ratio, val_ratio = 0.6, 0.2
        train_data = data[:int(n * train_ratio)]
        val_data   = data[int(n * train_ratio): int(n * (train_ratio + val_ratio))]
        test_data  = data[int(n * (train_ratio + val_ratio)):]

        total_data = [train_data, val_data, test_data]
        cur_data   = total_data[self.set_type]      # [T', N, 1]

        if self.scale:
            self.scaler.fit(train_data[:, :, 0])
            cur_data[:, :, 0] = self.scaler.transform(cur_data[:, :, 0])

        self.data_x = cur_data
        self.data_y = cur_data

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.input_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end, :, 0]
        seq_y = self.data_y[r_begin:r_end, :, 0:1]

        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.input_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_PEMS(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='M',
                 data_path='PEMS04.npz', target='OT', scale=True):
        self.input_len = size[0]
        self.pred_len = size[1]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        data = np.load(os.path.join(self.root_path, self.data_path))['data']

        train_ratio = 0.6
        valid_ratio = 0.2
        train_data = data[:int(train_ratio * len(data))]
        valid_data = data[int(train_ratio * len(data)): int((train_ratio + valid_ratio) * len(data))]
        test_data = data[int((train_ratio + valid_ratio) * len(data)):]

        total_data = [train_data, valid_data, test_data]
        cur_data = total_data[self.set_type]

        if self.scale:
            self.scaler.fit(train_data[:, :, 0])
            cur_data[:, :, 0] = self.scaler.transform(cur_data[:, :, 0])

        self.data_x = cur_data
        self.data_y = cur_data

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.input_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end, :, 0]
        seq_y = self.data_y[r_begin:r_end, :, 0:1]

        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.input_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)