import scipy.io as scio
import numpy as np
import random

from torch.utils.data import Dataset
import torch


class BpDataset(Dataset):
    def __init__(self, train: str, K=5):
        data_path = './data/bp/{}.npy'
        # view1 = torch.from_numpy(np.load(data_path.format('view1_train')))
        # view2 = torch.from_numpy(np.load(data_path.format('view2_train')))
        # label = torch.from_numpy(np.load(data_path.format('label_train')))
        # data_length = len(view1)
        # every_k_len = data_length // K
        if train == 'train':
            self.x_data_1 = torch.from_numpy(np.load(data_path.format('view1_train')))
            self.x_data_2 = torch.from_numpy(np.load(data_path.format('view2_train')))
            self.y_data = torch.from_numpy(np.load(data_path.format('label_train')))
        elif train == 'test':
            self.x_data_1 = torch.from_numpy(np.load(data_path.format('view1_test')))
            self.x_data_2 = torch.from_numpy(np.load(data_path.format('view2_test')))
            self.y_data = torch.from_numpy(np.load(data_path.format('label_test')))

            # self.x_data_1 = torch.cat([view1[: every_k_len * ki], view1[every_k_len * (ki+1):]])
            # self.x_data_2 = torch.cat([view2[: every_k_len * ki], view2[every_k_len * (ki+1):]])
            # self.y_data = torch.cat([label[: every_k_len * ki], label[every_k_len * (ki+1):]])
        # elif train == 'val':
        #     self.x_data_1 = torch.from_numpy(np.load(data_path.format('view1_valid')))
        #     self.x_data_2 = torch.from_numpy(np.load(data_path.format('view2_valid')))
        #     self.y_data = torch.from_numpy(np.load(data_path.format('label_valid')))
            # self.x_data_1 = view1[every_k_len * ki : every_k_len * (ki+1)]
            # self.x_data_2 = view2[every_k_len * ki : every_k_len * (ki+1)]
            # self.y_data = label[every_k_len * ki : every_k_len * (ki+1)]
        # elif train == 'test1':
        #     self.x_data_1 = torch.from_numpy(np.load(data_path.format('view1_test1')))
        #     self.x_data_2 = torch.from_numpy(np.load(data_path.format('view2_test1')))
        #     self.y_data = torch.from_numpy(np.load(data_path.format('label_test1')))
        # elif train == 'test2':
        #     self.x_data_1 = torch.from_numpy(np.load(data_path.format('view1_test2')))
        #     self.x_data_2 = torch.from_numpy(np.load(data_path.format('view2_test2')))
        #     self.y_data = torch.from_numpy(np.load(data_path.format('label_test2')))

    def __getitem__(self, item):
        return self.x_data_1[item], self.x_data_2[item], self.y_data[item]

    def __len__(self):
        return len(self.x_data_1)
    

class HivDataset(Dataset):
    def __init__(self, train: str, K=5):
        data_path = './data/hiv/{}.npy'
        if train == 'train':
            self.x_data_1 = torch.from_numpy(np.load(data_path.format('view1_train')))
            self.x_data_2 = torch.from_numpy(np.load(data_path.format('view2_train')))
            self.y_data = torch.from_numpy(np.load(data_path.format('label_train')))
        elif train == 'test':
            self.x_data_1 = torch.from_numpy(np.load(data_path.format('view1_test')))
            self.x_data_2 = torch.from_numpy(np.load(data_path.format('view2_test')))
            self.y_data = torch.from_numpy(np.load(data_path.format('label_test')))

    def __getitem__(self, item):
        return self.x_data_1[item], self.x_data_2[item], self.y_data[item]

    def __len__(self):
        return len(self.x_data_1)


class PpmiDataset(Dataset):
    def __init__(self, train: str, K=5):
        data_path = './data/ppmi/{}.npy'
        if train == 'train':
            self.x_data_1 = torch.from_numpy(np.load(data_path.format('view1_train')))
            self.x_data_2 = torch.from_numpy(np.load(data_path.format('view2_train')))
            self.y_data = torch.from_numpy(np.load(data_path.format('label_train')))
        elif train == 'test':
            self.x_data_1 = torch.from_numpy(np.load(data_path.format('view1_test')))
            self.x_data_2 = torch.from_numpy(np.load(data_path.format('view2_test')))
            self.y_data = torch.from_numpy(np.load(data_path.format('label_test')))

    def __getitem__(self, item):
        return self.x_data_1[item], self.x_data_2[item], self.y_data[item]

    def __len__(self):
        return len(self.x_data_1)


class Ppmi622Dataset(Dataset):
    def __init__(self, train: str, K=5):
        data_path = './data/ppmi_622/{}.npy'
        if train == 'train':
            self.x_data_1 = torch.from_numpy(np.load(data_path.format('view1_train')))
            self.x_data_2 = torch.from_numpy(np.load(data_path.format('view2_train')))
            self.y_data = torch.from_numpy(np.load(data_path.format('label_train')))
        elif train == 'test1':
            self.x_data_1 = torch.from_numpy(np.load(data_path.format('view1_test1')))
            self.x_data_2 = torch.from_numpy(np.load(data_path.format('view2_test1')))
            self.y_data = torch.from_numpy(np.load(data_path.format('label_test1')))
        elif train == 'test2':
            self.x_data_1 = torch.from_numpy(np.load(data_path.format('view1_test2')))
            self.x_data_2 = torch.from_numpy(np.load(data_path.format('view2_test2')))
            self.y_data = torch.from_numpy(np.load(data_path.format('label_test2')))

    def __getitem__(self, item):
        return self.x_data_1[item], self.x_data_2[item], self.y_data[item]

    def __len__(self):
        return len(self.x_data_1)