import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from resnet import resnet18
from dataset import Batch

from typing import Dict, Union, Tuple 

import logging

class Model(nn.Module):
    def __init__(
        self,
        dim: int,
        alpha: float,
        beta: float,
        device: Union[str, torch.device],
    ) -> None:
        super(Model, self).__init__()

        self._backbone = resnet18(dim)
        self._alpha = alpha
        self._beta = beta
        self._device = device

        self._memory_x = []
        self._memory_y = []
        self._minterms = None
        self._minterm_vecs = None

    def forget(self) -> None:
        self._memory_x = []
        self._memory_y = []
        self._minterms = None
        self._minterm_vecs = None
        logging.info("Memory reset")

    def remember(self, x: Tensor, data: Batch) -> None:
        self._memory_x.append(x.detach().cpu())
        self._memory_y.append(data.labels.cpu())

    def embed(self, data: Batch) -> Tensor:
        return self._backbone(data.images)

    def forward(self, data: Batch) -> Dict[str, Tensor]:
        x = self.embed(data)

        self.remember(x, data)

        z = torch.cat((data.labels.permute(1,0), 
                       x.permute(1,0)), dim=0)
        
        z_svals = torch.linalg.svdvals(z)
        x_svals = torch.linalg.svdvals(x)

        loss = z_svals.sum()
        loss -= self._alpha * x_svals.sum()
        loss += self._beta * x_svals.max() ** 2

        output = {"x_svals" : x_svals,
                  "z_svals" : z_svals,
                  "loss" : loss}

        return output

    def update_minterms(self) -> None:
        if len(self._memory_x) == 0:
            return None

        memory_x = torch.cat(self._memory_x, dim=0).to(self._device)
        memory_y = torch.cat(self._memory_y, dim=0).to(self._device)

        self._minterms = torch.unique(memory_y, dim=0)

        minterm_vecs = []

        for minterm in self._minterms:
            mask = (memory_y == minterm).all(dim=-1)
            selected_embeddings = memory_x[mask, :]
            U, _, _ = torch.linalg.svd(selected_embeddings.T)
            minterm_vecs.append(U[:,:1].T)

        self._minterm_vecs = torch.cat(minterm_vecs, dim=0)

    def evaluate(self, data: Batch) -> Tuple[Tensor, Tensor]:        
        batch_size = data.images.shape[0]

        self.update_minterms()

        if self._minterms is None:
            return torch.tensor(0.0), None

        x = self.embed(data)
        x = F.normalize(x, dim=-1, p=2)

        minterm_labels = torch.full((batch_size,), -1.0, device=self._device)

        logging.info(f"Evaluating with {len(self._minterms)} minterms")

        for i, minterm in enumerate(self._minterms):
            mask = (data.labels == minterm).all(dim=-1)
            minterm_labels.masked_fill_(mask, i)

        p = torch.square(x @ self._minterm_vecs.T)

        predictions = torch.argmax(p, dim=-1)

        return predictions[minterm_labels > -1], minterm_labels[minterm_labels > -1]


def compute_target_sigma(y_svals,
                         alpha: float,
                         beta: float):
    """
        Computes target singular value for the 
        minimization of the nuclear loss
    """
    def curve(x):
        x = x[0]
        return [np.array([x / np.sqrt(m**2 + x**2) for m in y_svals]).sum() - len(y_svals)*alpha + 2.0 * beta * x]

    def grad(x):
        x = x[0]
        return [np.array([m**2 / np.sqrt(m**2 + x**2)**3 for m in y_svals]).sum() + 2.0 * beta]

    sol = optimize.root(curve, [0.0], jac=grad, method='hybr')
    
    return sol.x[0]


