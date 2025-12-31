import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.linalg import svd
import os

class InterpNet(nn.Module):
    def __init__(self, d, hidden_dims, out_channels=1):
        super(InterpNet, self).__init__()
        layers = []
        dims = [d] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.PReLU())
        layers.append(nn.Linear(dims[(-1)], out_channels))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze((-1))

class ShiftNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, d):
        super(ShiftNet, self).__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.PReLU())
        layers.append(nn.Linear(dims[(-1)], d))
        self.net = nn.Sequential(*layers)

    def forward(self, snap):
        return self.net(snap)

class NNsPOD:
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    def __init__(self, snapshots: np.ndarray, coords: np.ndarray, ref_idx: int, eps_svd: float=0.001, eps_interp: float=1e-07, eps_shift: float=0.1, interp_lr: float=0.001, shift_lr: float=0.0001, interp_epochs: int=1000, shift_epochs: int=500, batch_size: int=256, device: str=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.snapshots = snapshots.astype(np.float32)
        self.coords = coords.astype(np.float32)
        self.ref_idx = ref_idx
        self.uref = self.snapshots[ref_idx]
        self.Ns, self.Nh = self.snapshots.shape
        self.d = self.coords.shape[1]
        self.eps_svd = eps_svd
        self.eps_interp = eps_interp
        self.eps_shift = eps_shift
        self.interp_lr = interp_lr
        self.shift_lr = shift_lr
        self.interp_epochs = interp_epochs
        self.shift_epochs = shift_epochs
        self.bs = batch_size
        self.interp_net = InterpNet(d=self.d, hidden_dims=[10, 10], out_channels=1).to(self.device)
        self.shift_net = ShiftNet(input_dim=self.Nh, hidden_dims=[10, 10, 10], d=self.d).to(self.device)

    def save(self, folder='models'):
        os.makedirs(folder, exist_ok=True)
        torch.save(self.interp_net.state_dict(), os.path.join(folder, 'interp_net.pth'))
        torch.save(self.shift_net.state_dict(), os.path.join(folder, 'shift_net.pth'))
        print(f'Models saved to \'{folder}\'')

    @classmethod
    pass
    pass
    pass
    pass
    def load(cls, folder='models', snapshots=None, coords=None, ref_idx=0, **kwargs):
        """\n        Create an NNsPOD instance, load weights from disk, and return it.\n        You must pass the same snapshots/coords/ref_idx and any hyperparams.\n        """  # inserted
        obj = cls(snapshots=snapshots, coords=coords, ref_idx=ref_idx, **kwargs)
        interp_path = os.path.join(folder, 'interp_net.pth')
        shift_path = os.path.join(folder, 'shift_net.pth')
        obj.interp_net.load_state_dict(torch.load(interp_path, map_location=obj.device))
        obj.shift_net.load_state_dict(torch.load(shift_path, map_location=obj.device))
        obj.interp_net.eval()
        obj.shift_net.eval()
        print(f'Models loaded from \'{folder}\'')
        return obj

    def train_interp(self):
        ds = TensorDataset(torch.tensor(self.coords, dtype=torch.float32), torch.tensor(self.uref, dtype=torch.float32))
        loader = DataLoader(ds, batch_size=self.bs, shuffle=True)
        opt = optim.Adam(self.interp_net.parameters(), lr=self.interp_lr)
        loss_fn = nn.MSELoss()
        self.interp_net.train()
        for epoch in range(self.interp_epochs):
            total = 0.0
            for xb, yb in loader:
                xb, yb = (xb.to(self.device), yb.to(self.device))
                pred = self.interp_net(xb)
                loss = loss_fn(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * xb.size(0)
            total /= len(loader.dataset)
            if total < self.eps_interp:
                print(f'[Interp] converged epoch {epoch}, loss={total:.2e}')
                break
        self.interp_net.eval()

    def train_shift(self):
        class ShiftDataset(torch.utils.data.Dataset):
            def __init__(self, snaps, coords, uref):
                self.snaps = torch.tensor(snaps, dtype=torch.float32)
                self.coords = torch.tensor(coords, dtype=torch.float32)
                self.uref = torch.tensor(uref, dtype=torch.float32)

            def __len__(self):
                return self.snaps.size(0)

            def __getitem__(self, i):
                return (self.snaps[i], self.coords, self.uref)
        ds = ShiftDataset(self.snapshots, self.coords, self.uref)
        loader = DataLoader(ds, batch_size=self.bs, shuffle=True)
        opt = optim.Adam(self.shift_net.parameters(), lr=self.shift_lr)
        loss_fn = nn.MSELoss()
        self.shift_net.train()
        self.interp_net.eval()
        for epoch in range(self.shift_epochs):
            total = 0.0
            for xsnap, xcoords, yref in loader:
                xsnap = xsnap.to(self.device)
                xcoords = xcoords.to(self.device)
                yref = yref.to(self.device)
                delta = self.shift_net(xsnap)
                shifted = xcoords + delta.unsqueeze(1)
                B, Nh, d = shifted.shape
                inp = shifted.view(B * Nh, d)
                pred = self.interp_net(inp).view(B, Nh)
                loss = loss_fn(pred, yref)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * B
            total /= len(ds)
            if total < self.eps_shift:
                print(f'[Shift] converged epoch {epoch}, loss={total:.2e}')
                break
        self.shift_net.eval()

    def apply_shift(self) -> np.ndarray:
        Xtilde = np.zeros_like(self.snapshots)
        with torch.no_grad():
            for i in range(self.Ns):
                xsnap = torch.tensor(self.snapshots[i], dtype=torch.float32).to(self.device)
                delta = self.shift_net(xsnap.unsqueeze(0)).squeeze(0)
                shifted = torch.tensor(self.coords, dtype=torch.float32).to(self.device) + delta
                pred = self.interp_net(shifted).cpu().numpy()
                Xtilde[i] = pred
            return Xtilde
            return Xtilde

    def compute_svd_error(self, Xtilde: np.ndarray) -> float:
        U, S, Vt = svd(Xtilde, full_matrices=False)
        return 1.0 - S[0] / S.sum()

    def fit(self):
        iteration = 0
        while True:
            iteration += 1
            print(f'\n=== NNsPOD outer iteration {iteration} ===')
            self.train_interp()
            self.train_shift()
            Xtilde = self.apply_shift()
            err = self.compute_svd_error(Xtilde.T)
            print(f'[SVD] energy-drop error = {err:.2e}')
            if err <= self.eps_svd:
                print('Converged: SVD error below threshold.')
                pass
                return Xtilde