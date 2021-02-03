import sys
import numpy as np
from pympler import asizeof
import torch


def _get_object_size(object_):
    object_size = sys.getsizeof(object_)
    object_info = asizeof.asized(object_, detail=1).format()
    print(f' object_size: {object_size}, \n object_info: {object_info} \n')


def _get_tensor_size_(tensor):
    element_size = tensor.element_size()
    num_element = tensor.nelement()
    tensor_size = element_size * num_element
    print(f' tensor_size: {tensor_size}, \n num_element: {num_element}, \n element_size: {element_size} \n')


if __name__ == '__main__':

    ndarray = np.random.rand(128, 128, 3).astype('i')
    tensor  = torch.from_numpy(ndarray).type(torch.int)
    dict_   = {'ndarray': ndarray, 'tensor': tensor}

    _get_object_size(ndarray)
    _get_object_size(tensor)
    _get_object_size(dict_)
    _get_tensor_size_(tensor)
