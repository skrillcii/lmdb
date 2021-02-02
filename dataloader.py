import sys
import tqdm
import numpy as np
from PIL import Image
from pympler import asizeof

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

import lmdb_


class MNIST(Dataset):

    def __init__(self, root, set_, lmdb):

        self.root  = root
        self.lmdb  = lmdb

        ds = torchvision.datasets.MNIST(root, train=set_, download=True)
        data = [np.transpose(np.tile(np.array(ds[i][0].resize((128, 128), Image.NEAREST)),
                   [3, 1, 1]), [1, 2, 0]) for i in range(len(ds))]
        label = [ds[i][1] for i in range(len(ds))]

        # import ipdb
        # ipdb.set_trace()

        if lmdb:
            dummy = {'data': data[0], 'label': label[0]}
            # dummy_size = sys.getsizeof(dummy)
            # dummy_info = asizeof.asized(dummy, detail=1).format()
            dummy_size = 49784
            buffer_size = 10
            map_size = dummy_size * buffer_size * len(data)
            for i in tqdm.tqdm(range(len(data))):
                object_ = {'data': data[i], 'label': label[i]}
                lmdb_.store(root, object_, i, map_size)
        else:
            self.data = data
            self.label = label


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if lmdb:
            cur_data = lmdb_.read(self.root, idx)
            return cur_data
        else:
            cur_data = self.data[idx]
            cur_label = self.label[idx]
            return {'data': cur_data, 'label': cur_label}
