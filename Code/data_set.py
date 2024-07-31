import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import img_pro


class set(Dataset):
    # i_dir The folder path where the original image is located
    # c_dir The folder path where the crater label is located
    # t_dit The folder path where the ejecta label is located
    # d_dir a txt file path with all file names
    def __init__(self, i_dir, c_dir, t_dir, d_dir, net = None):
        self.i_dir = i_dir
        self.t_dir = t_dir
        self.c_dir = c_dir
        self.d_dir = d_dir
        self.net = net
        with open(d_dir, 'r') as f:
            self.data_name_list = f.readlines()
        for i in range(len(self.data_name_list)):
            self.data_name_list[i] = self.data_name_list[i].strip()

    def __getitem__(self, i):
        x1 = np.array(img_pro.resize(Image.open(os.path.join(self.i_dir, self.data_name_list[i] + '.jpg'))), np.float32)
        x2 = np.array(img_pro.resize(Image.open(os.path.join(self.c_dir, self.data_name_list[i] + '.png'))), np.float32)
        x1 = np.transpose(x1, [2, 0, 1]) / 255.
        x2 = np.expand_dims(x2, 0)
        x = np.concatenate([x1, x2], 0)
        x = torch.tensor(x)
        y = np.array(img_pro.resize(Image.open(os.path.join(self.t_dir, self.data_name_list[i] + '.png'))), np.float32)
        y = torch.from_numpy(y).to(torch.int64)
        return x, y, self.data_name_list[i]

    def __len__(self):
            return len(self.data_name_list)

    def cal_miou(self):
        pass

    def cal_loss(self):
        pass

    def cal_con_mat(self):
        pass

