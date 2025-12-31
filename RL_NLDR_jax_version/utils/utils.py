
import numpy as np
import random
from typing import Tuple
import matplotlib.pyplot as plt

def random_selection_arr_maker(k: int, l: int) -> np.array:
    
    """
    Input:
    k -> Length of array needed.
    l -> Number of 1s in the array.
    Output:
    selection_arr: Array of length 'k' with 'l' 1s at randomly chosen positions.
    """

    selection_arr = np.zeros(shape=(k,), dtype=int)
    indices_chosen = random.sample(range(len(selection_arr)), l)
    for i in indices_chosen:
        selection_arr[i] = 1
    return selection_arr

def apply_selected_funcs(S_hat, lib_funcs, sel_arr):
    """
    Select functions from 'lib_funcs' based on indices in 'sel_arr' that have 1. Pass S_hat
    through those functions to get S_mod.
    """
    selected_funcs = [eval(f'lambda _: {func}') for func, sel in zip(lib_funcs, sel_arr) if sel == 1]
    results = [func(S_hat) for func in selected_funcs]
    return np.concatenate(results, axis=0)

def compute_reconstr_error(S, S_ref, U_trunc, S_hat, library_functions, selection_arr) -> Tuple[np.array, float]:
    """
    Compute reconstruction error.
    Input: S, S_ref, U_trunc, 
           S_hat, library functions, selection array
    
    Output: V_bar, reconstr_err in
    S = S_ref + U_trunc@S_hat + V_bar@S_nl
    
    S_nl is obtained as: S_hat -> selected library_functions -> selection_arr[len(library_functions): ] 
    """
    S_mod = apply_selected_funcs(S_hat, library_functions, selection_arr[:len(library_functions)])
    S_nl = S_mod[selection_arr[len(library_functions):].astype('bool')]
    S_lin_reconstr = U_trunc @ S_hat
    V_bar = (S - S_ref - S_lin_reconstr) @ np.linalg.pinv(S_nl)
    reconstr_S = S_ref + S_lin_reconstr + V_bar @ S_nl
    reconstr_err = np.linalg.norm(S - reconstr_S, 'fro') / np.linalg.norm(S, 'fro')
    return (V_bar, reconstr_err)

def visual(batch_rewards: np.array) -> None:
    """
    visualize batch rewards in training.
    """
    plt.plot(batch_rewards)
    plt.xlabel('batch index')
    plt.ylabel('batch reward value')
    plt.tight_layout()
    plt.savefig('batch_reward_plot')