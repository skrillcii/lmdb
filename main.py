import argparse

import torch
import torchvision
from torch.utils.data import DataLoader

from dataloader import MNIST


def main(args):

    import ipdb
    ipdb.set_trace()

    ds_tr = MNIST(args.ds_root, set_=True, lmdb=args.lmdb)
    ds_te = MNIST(args.ds_root, set_=False, lmdb=args.lmdb)

    dl_kargs = {
        'batch_size': 128,
        'num_workers': 4,
        'pin_memory': True,
        'drop_last': False,
    }
    dl_tr = DataLoader(ds_tr, shuffle=True, **dl_kargs)
    dl_te = DataLoader(ds_te, shuffle=False, **dl_kargs)

    for cur_data in enumerate(dl_tr):
        print(cur_data)
    for cur_data in enumerate(dl_te):
        print(cur_data)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_root', default='./dataset', type=str)
    parser.add_argument('--lmdb', default=True, type=bool)
    args = parser.parse_args()

    main(args)
