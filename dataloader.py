import sys
import tqdm
import lmdb
import pickle
import numpy as np
from PIL import Image
from pympler import asizeof

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader


class MNIST(Dataset):

    def __init__(self, root, set_, lmdb_):

        self.root    = root
        self.data    = None
        self.label   = None
        self.lmdb_io = None

        num_samples = 1
        dataset = torchvision.datasets.MNIST(root, train=set_, download=True)
        data = [np.transpose(np.tile(np.array(dataset[i][0].resize((128, 128), Image.NEAREST)),
                                     [3, 1, 1]), [1, 2, 0]) for i in range(num_samples)]
        label = [dataset[i][1] for i in range(num_samples)]

        if lmdb_:
            dummy        = {'data': data[0], 'label': label[0]}
            dummy_tensor = {'data': torch.from_numpy(data[0]), 'label': torch.tensor(label[0])}
            dummy_size   = 49152
            buffer_size  = 3
            map_size     = dummy_size * buffer_size * len(data)
            self.lmdb_io = {
                    'writer': lmdb.open(root+'/lmdb', map_size=map_size, max_readers=128,
                        readonly=False, lock=True, writemap=False),
                    'reader': lmdb.open(root+'/lmdb', map_size=map_size, max_readers=128,
                        readonly=True, lock=False)
            }

            # Check memory sizes
            self._get_object_size(dummy)
            self._get_object_size(dummy_tensor)
            self._get_tensor_size_(dummy_tensor['data'])
            self._get_tensor_size_(dummy_tensor['label'])

        self.data  = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        cur_sample = None
        if self.lmdb_io is not None:
            with self.lmdb_io['reader'].begin(write=False) as txn:
                cur_sample = txn.get(('idx-{:08d}'.format(idx)).encode('ascii'))
        if cur_sample is not None:
            cur_sample = pickle.loads(cur_sample)
        else:
            cur_sample = {'data': torch.from_numpy(self.data[idx]), 'label': torch.tensor(self.label[idx])}
            with self.lmdb_io['writer'].begin(write=True) as txn:
                txn.put(f'idx-{idx}'.encode('ascii'), pickle.dumps(cur_sample))
            return cur_sample

    @staticmethod
    def _get_object_size(object_):
        object_size = sys.getsizeof(object_)
        object_info = asizeof.asized(object_, detail=1).format()
        print(f' object_size: {object_size}, \n object_info: {object_info} \n')

    @staticmethod
    def _get_tensor_size_ (tensor):
        element_size = tensor.element_size()
        num_element  = tensor.nelement()
        tensor_size  = element_size * num_element
        print(f' tensor_size: {tensor_size}, \n num_element: {num_element}, \n element_size: {element_size} \n')
