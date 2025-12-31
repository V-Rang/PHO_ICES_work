
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from utils import random_selection_arr_maker

class SelectionDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.tensor(self.data[index], dtype=torch.int32)

def create_input(num_library_functions, num_library_functions_select, selection_length, sub_selection_length, trunc_svd_shape, num_samples_create):
    """
    Input:
    num_library_functions: Total number of functions in the library.
    num_library_functions_select: Number of library functions to select.
    selection_length: array size of each selection array.
    sub_selection_length: number of 1s in each selection array. 
    trunc_svd_shape: Shape of S_hat
    num_samples_create: Total number of samples to create for the Torch Dataset.
    
    trunc_svd_shape*num_library_functions_select = shape of modified array.
    
    Output:
    A Torch dataset that can be used to create a DataLoader to iterate through.
    """
    all_samples = []
    new_shape = trunc_svd_shape * num_library_functions_select
    num_networks_type_2 = new_shape % selection_length
    num_networks_type_3 = new_shape // selection_length - num_networks_type_2
    for i in range(num_samples_create):
        lib_selection_arr = random_selection_arr_maker(num_library_functions, num_library_functions_select)
        selection_arr_type_2 = np.array([])
        for j in range(num_networks_type_2):
            selection_arr_type_2 = np.concatenate((selection_arr_type_2, random_selection_arr_maker(selection_length + 1, sub_selection_length)), axis=0)
        selection_arr_type_3 = np.array([])
        for j in range(num_networks_type_3):
            selection_arr_type_3 = np.concatenate((selection_arr_type_3, random_selection_arr_maker(selection_length, sub_selection_length)), axis=0)
        sample_array = np.concatenate((lib_selection_arr, selection_arr_type_2, selection_arr_type_3))
        all_samples.append(sample_array)
    return SelectionDataset(all_samples)