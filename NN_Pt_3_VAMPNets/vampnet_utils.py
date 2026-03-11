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
        cov_00 = 1/(x.shape[0] - 1) * torch.matmul(x, x.T)
        cov_01 = 1/(x.shape[0] - 1) * torch.matmul(x, y.T)
        cov_11 = 1/(x.shape[0] - 1) * torch.matmul(y, y.T)
        vamp_matrix = torch.sqrt(torch.linalg.solve(
            torch.sqrt(torch.linalg.solve(cov_00, cov_01, left = True)), cov_11, left = False))
        return -1*torch.square(torch.linalg.matrix_norm(vamp_matrix))