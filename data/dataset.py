import readjson
import os
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data_path):
        super(MyDataset, self).__init__()
        self.samples = []

        with open(data_path, 'r') as f:
            js = json.load(f)

        # self.min = js['min']
        # self.max = js['max']

        for key, val in js.items():
            if key not in ['0', '1', '2', '3', '4']:
                continue
            label = key
            for feature in val:
                self.samples.append((feature, label))
        print('have {} data'.format(len(self.samples)))

    def transform(self, vec):
        vec = torch.Tensor(vec)
        vec = torch.unsqueeze(torch.unsqueeze(vec, 0), 0)
        vec = vec.permute(2, 0 ,1)
        return vec

    def __getitem__(self, index):
        feature, label = self.samples[index]

        feature = self.transform(feature)
        label = int(label)

        return feature, label

    def __len__(self):
        return len(self.samples)

    def public_method(self, index):
        return self.__getitem__(index)

if __name__ == '__main__':
    data_path = '/home/zhaomengjun/2021_binglang_paper/feature_dict/test_data.json'
    dataset = MyDataset(data_path)
    feature, label = dataset.public_method(0)