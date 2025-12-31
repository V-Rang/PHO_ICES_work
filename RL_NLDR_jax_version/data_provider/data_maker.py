
import numpy as np
from .dataset_maker import DatasetCreate
from torch.utils.data import DataLoader
from utils.tools import random_selection_arr_maker

def data_maker(args):
    sample_selection_arrays = []
    sample_length = args.trunc_dim * len(args.library_functions)
    for i in range(args.num_samples_total):
        sample_arr = np.array([])
        for j in range(sample_length // args.selection_length):
            sample_arr = np.concatenate((sample_arr, random_selection_arr_maker(args.selection_length, args.sub_selection_length)), axis=0)
        sample_selection_arrays.append(sample_arr)
    sample_selection_arrays = np.array(sample_selection_arrays)
    train_dataset, train_dataloader = train_data_maker(sample_selection_arrays, args.batch_size, args.shuffle_flag, args.drop_last)
    return (sample_selection_arrays, train_dataset, train_dataloader)

def train_data_maker(train_data, batch_size, shuffle_flag, drop_last):
    """
    Input: dict of parameters, flag for train data.
    Output: Torch dataset and dataloader.
    """
    data_set = DatasetCreate(train_data)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle_flag, drop_last=drop_last)
    return (data_set, data_loader)