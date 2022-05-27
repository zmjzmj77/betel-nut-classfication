import json
import os
import cv2
from torch.utils.data import Dataset
from data.transform import transforms
import random
import numpy as np

class res_Dataset(Dataset):
    def __init__(self, root, dataaug = False):
        self.samples = []

        with open(root, 'r') as f:
            js = json.load(f)

        for type, path_list in js.items():
            for path in path_list:
                self.samples.append([path, int(type) - 4])

        self.transform = transforms()
        self.data_len = len(self.samples)
        self.aug = dataaug

    def __getitem__(self, index):
        img_path, label = self.samples[index]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.samples)