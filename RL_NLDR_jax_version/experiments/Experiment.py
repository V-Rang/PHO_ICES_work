
from models import model_1
from data_provider.data_maker import data_maker
import numpy as np
import torch
from tqdm import tqdm
import os
import pickle

class Experiment:

    def __init__(self, args) -> None:
        assert args.trunc_dim * len(args.library_functions) % args.selection_length == 0
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_dict = {'model_1': model_1}
        model_params = {'library_functions': self.args.library_functions, 'gamma': self.args.gamma, 'S_train': self.args.S_train, 'S_train_dot': self.args.S_train_dot, 'S_train_ref': self.args.S_train_ref, 'lam1_vec': self.args.lam1_vec, 'lam2_vec': self.args.lam2_vec, 'trunc_dim': self.args.trunc_dim, 'selection_length': self.args.selection_length, 'sub_selection_length': self.args.sub_selection_length, 'd_model': self.args.d_model, 'e_layers': self.args.e_layers, 'learning_rate': self.args.learning_rate, 'batch_size': self.args.batch_size, 't_vals_train': self.args.t_vals_train}
        self.model = self.model_dict[self.args.model_type].Model(model_params)
        self.model = self.model.to(self.device)

    def _get_data(self):
        data, data_set, data_loader = data_maker(self.args)
        return (data, data_set, data_loader)

    def train(self, path):
        learning_rate_batch_sample_rewards_filename = path + self.args.lr_sample_rewards_filename
        if not os.path.exists(path):
            os.makedirs(path)
        lrs_sample_rewards = {}
        lrs_sample_rewards['modified_learning_rates'] = []
        lrs_sample_rewards['sample_rewards_batch'] = []
        _, _, train_dataloader = self._get_data()
        best_sample_list = []
        batch_reward_list = []
        for epoch in tqdm(range(self.args.num_epochs)):
            best_batch_reward = -np.inf
            for i, batch_sel_arrs in enumerate(train_dataloader):
                batch_sel_arrs = batch_sel_arrs.to(device=self.device, dtype=torch.float32)
                batch_grads, batch_reward, best_sample_batch, batch_sample_rewards, num_samples_processed = self.model(batch_sel_arrs)
                print(num_samples_processed)
                if num_samples_processed == 0:
                    continue
                batch_reward_list.append(batch_reward)
                if batch_reward > best_batch_reward:
                    best_batch_reward = batch_reward
                    best_sample_epoch = best_sample_batch
                lr = self.model.update_grads(batch_grads, path, batch_sample_rewards, num_samples_processed)
                lrs_sample_rewards['modified_learning_rates'].append(lr)
                lrs_sample_rewards['sample_rewards_batch'].append(batch_sample_rewards)
            best_sample_list.append(best_sample_epoch)
        return (batch_reward_list, best_sample_list, lrs_sample_rewards)