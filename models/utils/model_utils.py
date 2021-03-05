import json
import os
from torchvision import datasets, transforms
from collections import defaultdict
import random
import numpy as np
import torch
from .so_tag_utils import *

VOCAB_DIR = 0
emb_array = 0
vocab = 0
embed_dim = 0


def batch_data(data, batch_size, rng=None, shuffle=True, eval_mode=False, full=False, malicious=False):
    """
    data is a dict := {'x': [list], 'y': [list]} with optional fields 'y_true': [list], 'x_true' : [list]
    If eval_mode, use 'x_true' and 'y_true' instead of 'x' and 'y', if such fields exist
    returns x, y, which are both lists of size-batch_size lists
    """
    x = data['x_true'] if eval_mode and 'x_true' in data else data['x']
    y = data['y_true'] if eval_mode and 'y_true' in data else data['y']
    if malicious:
        y = np.random.randint(0,500,len(y))
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    x = torch.tensor(x).cuda()
    y = torch.LongTensor(y).cuda()
    raw_x = x[indices]
    raw_y = y[indices]
    batched_x, batched_y = [], []
    if not full:
        for i in range(0, len(raw_x), batch_size):
            batched_x.append(raw_x[i:i + batch_size])
            batched_y.append(raw_y[i:i + batch_size])
    else:
        batched_x.append(raw_x)
        batched_y.append(raw_y)
    return batched_x, batched_y


def read_so_data():
    groups = []
    train_data, test_data = get_centralized_datasets()
    clients = {
        'train_users': list(train_data.keys()),
        'test_users': list(test_data.keys())
    }
    return clients, groups, train_data, test_data





