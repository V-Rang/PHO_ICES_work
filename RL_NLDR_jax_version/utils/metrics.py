
import numpy as np
from utils.tools import apply_selected_funcs

def reconstruction_error(S, best_sample):
    """
    Use the function from 'tools.py' since this one involves repeated 
    computation of U_trunc, S_hat.

    Function to return linear and non-linear reconstruction errors.
    S = S_ref + U@S_hat + V_bar@S_nl
    Linear error =  ||(S - S_ref) - U@S_hat||_{F}/ ||S - S_ref||_{F}
    Non-linear error =  ||(S - S_ref) - U@S_hat - V_bar@S_nl||_{F}/ ||S - S_ref||_{F}
    """
    S_ref = np.mean(S, axis=1).reshape(-1, 1)
    S_org = S - S_ref
    S_norm = np.linalg.norm(S_org, 'fro')
    U_trunc = best_sample['U_trunc']
    library_functions = best_sample['library_functions']
    V_bar = best_sample['sample_V_bar']
    selection_arr = best_sample['selection_arr']
    S_hat = U_trunc.T @ (S - S_ref)
    S_org = S - S_ref
    S_norm = np.linalg.norm(S_org, 'fro')
    S_reconstr_linear = U_trunc @ S_hat
    linear_error = np.linalg.norm(S_org - S_reconstr_linear, 'fro') / S_norm
    S_mod = apply_selected_funcs(S_hat, library_functions, selection_arr[:len(library_functions)])
    S_nl = S_mod[selection_arr[len(library_functions):].detach().numpy().astype('bool')]
    S_reconstr_nl = S_reconstr_linear + V_bar @ S_nl
    nonlinear_error = np.linalg.norm(S_org - S_reconstr_nl, 'fro') / S_norm
    reconstruction_errors = {'linear_error': linear_error, 'nonlinear_error': nonlinear_error}
    return reconstruction_errors