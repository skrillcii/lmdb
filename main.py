import sys
import argparse
from pympler import asizeof

import torch
import torchvision
from torch.utils.data import DataLoader

from dataloader import MNIST


def main(args):

    ds_tr = MNIST(args.ds_root, set_=True, lmdb_=args.lmdb)
    ds_te = MNIST(args.ds_root, set_=False, lmdb_=args.lmdb)

    dl_kargs = {
        'batch_size': 1,
        'num_workers': 0,
        'pin_memory': True,
        'drop_last': False,
    }
    dl_tr = DataLoader(ds_tr, shuffle=True, **dl_kargs)
    dl_te = DataLoader(ds_te, shuffle=False, **dl_kargs)

    for cur_data in enumerate(dl_tr):
        data_size = sys.getsizeof(cur_data)
        data_info = asizeof.asized(cur_data, detail=1).format()
        print(data_size, data_info)
    for cur_data in enumerate(dl_te):
        data_size = sys.getsizeof(cur_data)
        data_info = asizeof.asized(cur_data, detail=1).format()
        print(data_size, data_info)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_root', default='./dataset', type=str)
    parser.add_argument('--lmdb', default=True, type=bool)
    args = parser.parse_args()
    main(args)
