import numpy as np
import torch
import torch.nn as nn

def com_center(coords: np.array, weights: np.array) -> np.array:
    """
    COM centers the input coordinate array
    Params:
        coords: np.array : input coordinates, as a [n_frames, n_atoms, dims]
                           or [n_atoms, dims] array
        weights: np.array : weights for each atom as an [n_atoms] array
    Returns:
        np.array : COM-centered weights array
    """
    if len(coords.shape) == 3:
        if weights:
            com = np.average(coords, axis = 1, weights = weights)
        else:
            com = np.average(coords, axis = 1)
        centered = []
        for frame, com_ in zip(coords, com):
            centered.append(frame - com_)
        return np.array(centered)
    elif len(coords.shape) == 2:
        if weights:
            com = np.average(coords, axis = 0, weights = weights)
        else:
            com = np.average(coords, axis = 1)
        frame = frame = com
        return frame
    
    else:
        raise ValueError('Input coords array must be of dim 2 or 3')
        

class VampNetLoss(nn.Module):
    def __init__(self):
        self.eps = 1e-10
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculates VAMP2 score from head outputs.
        Expects softmaxed outputs for x and y!
        Params:
            x: torch.Tensor : time t head output
            y: torch.Tensor : time t + tau head output
        Returns:
            torch.Tensor : VAMP2 loss
        """
        x, y = self._prep_data(x, y)
        cov_00 = 1/(x.shape[0] - 1) * torch.matmul(x, x.T)
        cov_01 = 1/(x.shape[0] - 1) * torch.matmul(x, y.T)
        cov_10 = 1/(x.shape[0] - 1) * torch.matmul(y, x.T)
        cov_11 = 1/(x.shape[0] - 1) * torch.matmul(y, y.T)
        auto_cov_inv = self._inv(1/2 * (cov_00 + cov_11), True)
        cross_cov = 1/2 * (cov_10 + cov_01)
        vamp_matrix = auto_cov_inv @ cross_cov @ auto_cov_inv
        return -1*torch.square(torch.linalg.matrix_norm(vamp_matrix))

    def _prep_data(self, x: torch.Tensor, y: torch.Tensor):
        x_t = x.T
        y_t = y.T
        x_ = x_t - torch.mean(x, dim = 1)
        y_ = y_t - torch.mean(y, dim = 1)
        return x_, y_

    def _inv(self, x: torch.Tensor, sqrt: bool):
        eigval_all, eigvec_all = torch.linalg.eigh(x)
        eigval_mask = eigval_all > self.eps
        eigval = eigval_all[eigval_mask]
        eigvec = eigvec_all[eigval_mask]
        if sqrt:
            eigval_inv= torch.diag(torch.sqrt(1/eigval))
        else:
            eigval_inv = torch.diag(1/eigval)
        x_inv = eigvec.T @ eigval_inv @ eigvec
        return x_inv