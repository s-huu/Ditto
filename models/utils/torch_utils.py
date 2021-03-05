import functools
import numpy as np
import torch


# Utilities to convert list of torch tensors into a single numpy array
def get_sizes(lst):
    sizes = []
    for w in lst:
        sizes.append(functools.reduce((lambda x, y: x*y), w.size()))
    c = np.cumsum(sizes)
    bounds = list(zip([0] + c[:-1].tolist(), c.tolist()))
    return sizes, bounds


def torch_to_numpy(lst, arr=None):
    # lst: obtained either from list(net.parameters()) or from torch.autograd.grad
    lst = list(lst)
    sizes, bounds = get_sizes(lst)
    if arr is None:
        arr = np.zeros(sum(sizes))
    else:
        assert len(arr) == sum(sizes)
    for bound, var in zip(bounds, lst):
        arr[bound[0]: bound[1]] = var.data.cpu().numpy().reshape(-1)
    return arr



def numpy_to_torch(arr, net):
    device = next(net.parameters()).device
    arr = torch.from_numpy(arr).to(device)
    sizes, bounds = get_sizes(net.trainable_parameters())
    assert len(arr) == sum(sizes)
    for bound, var in zip(bounds, net.trainable_parameters()):
        vnp = var.data.view(-1)
        vnp[:] = arr[bound[0] : bound[1]]
    return net
