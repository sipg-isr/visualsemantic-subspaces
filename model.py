import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.linalg import svdvals, eigvalsh, svd, eigh
from resnet import *
from convnet import ConvNet
from dataset import Batch

from typing import Dict, Union, Tuple, Callable, Optional

import logging

class Model(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        backbone: str,
        activation: Callable,
        alpha: float,
        beta: float,
        kernel: str,
        device: Union[str, torch.device],
    ) -> None:
        super(Model, self).__init__()

        self._dim_in = dim_in
        self._dim_out = dim_out
        self._alpha = alpha
        self._beta = beta
        self._kernel = kernel
        self._device = device

        if backbone == "resnet18":
            self._backbone = resnet18(
                in_dim=dim_in,
                out_dim=dim_out + int(kernel != "linear"),
                activation=activation
            )
        elif backbone == "resnet34":
            self._backbone = resnet34(
                in_dim=dim_in,
                out_dim=dim_out + int(kernel != "linear"),
                activation=activation
            )
        elif backbone == "convnet":
            self._backbone = ConvNet(
                in_dim=dim_in,
                out_dim=dim_out + int(kernel != "linear"),
                activation=activation
            )
        else:
            raise ValueError(f"Unknown backbone {backbone}")

        self._memory_x = []
        self._memory_y = []
        self._minterms = []
        self._minterm_evecs = []
        self._minterm_samples = []

    def forget(self) -> None:
        self._memory_x = []
        self._memory_y = []
        self._minterms = []
        self._minterm_evecs = []
        self._minterm_samples = []
        logging.info("Memory reset")

    def remember(self, x: Tensor, data: Batch) -> None:
        self._memory_x.append(x.detach().cpu())
        self._memory_y.append(data.labels.cpu())

    def embed(self, data: Batch) -> Tensor:
        feat = self._backbone(data.images)
        if self._kernel == "linear":
            return feat, None
        else:
            x, norm = torch.split(feat, (self._dim, 1), -1)
            return x, norm

    def kernel(self, x1: Tensor, x2: Tensor) -> Tensor:
        if self._kernel == "linear":
            return x1 @ x2.T
        elif self._kernel == "gaussian":
            return torch.exp(-(torch.cdist(x1, x2, p=2)**2))
        else:
            raise NotImplemented(f"Unknown kernel {self._kernel}")

    def forward(self, data: Batch) -> Dict[str, Tensor]:
        x, norm = self.embed(data)
        y = data.labels.float()

        if self._kernel == "linear":
            z = torch.cat((y.T, x.T), dim=0)
            z_svals = svdvals(z)
            x_svals = svdvals(x)
        else:
            kernel = norm.view(1,-1) * self.kernel(x,x) * norm.view(-1,1)
            z_svals = torch.sqrt(F.relu(eigvalsh(y @ y.T + kernel)))
            x_svals = torch.sqrt(F.relu(eigvalsh(kernel)))

        self.remember(x, data)

        loss = z_svals.sum()
        loss -= self._alpha * x_svals.sum()
        loss += self._beta * x_svals.max() ** 2

        output = {"x_norm" : x_svals.sum(),
                  "z_norm" : z_svals.sum(),
                  "loss" : loss}
        return output

    @torch.no_grad()
    def update_minterms(self) -> None:
        if len(self._memory_x) == 0:
            return None

        memory_x = torch.cat(self._memory_x, dim=0).to(self._device)
        memory_y = torch.cat(self._memory_y, dim=0).to(self._device)

        self._minterms = [] 
        self._minterm_evecs = []
        self._minterm_samples = []

        minterms = torch.unique(memory_y, dim=0)
        for minterm in minterms:
            mask = (memory_y == minterm).all(dim=-1)
            minterm_samples = memory_x[mask, :]
            
            if self._kernel == "linear":
                minterm_samples = F.normalize(minterm_samples, dim=-1, p=2)
                U, _, _ = svd(minterm_samples.T)
                self._minterm_evecs.append(U[:,:1].T)
                self._minterms.append(minterm)
                self._minterm_samples.append(minterm_samples)
            else:
                K = self.kernel(minterm_samples, minterm_samples)
                try:
                    lbds, U = eigh(K)
                except:
                    logging.error(f"eigh error minterm {minterm}")
                else:
                    self._minterms.append(minterm)

                    lbd_mask = (lbds / len(minterm_samples)) > 0.05
                    self._minterm_evecs.append(U[:,lbd_mask].T / torch.sqrt(lbds[lbd_mask].view(-1,1)))
                    self._minterm_samples.append(minterm_samples)

    @torch.no_grad()
    def evaluate(self, data: Batch) -> Tuple[Tensor, Tensor]:        
        batch_size = data.images.shape[0]

        if self._minterms is None:
            return torch.tensor([0]), torch.tensor([0])

        logging.info(f"Eval with {len(self._minterms)} minterms")

        # Create minterm labels (ficticious labels)
        minterm_labels = torch.full((batch_size,), -1.0, device=self._device)
        for i, minterm in enumerate(self._minterms):
            mask = (data.labels == minterm).all(dim=-1)
            minterm_labels.masked_fill_(mask, i)

        x_query, _ = self.embed(data)

        if self._kernel == "linear":
            minterm_evecs = torch.cat(self._minterm_evecs, dim=0)
            x_query = F.normalize(x_query, dim=-1, p=2)
            p = torch.square(x_query @ minterm_evecs.T)
        else:
            p = []
            for i in range(len(self._minterms)):
                # (batch_size, n_minterm_samples)
                kernel_memory_query = self.kernel(x_query, self._minterm_samples[i])

                # (batch_size, n_evecs)
                projections = torch.einsum(
                    "bj,kj->bk",
                    kernel_memory_query,
                    self._minterm_evecs[i]
                )
                
                p.append(torch.square(projections).sum(-1).view(-1,1))
            p = torch.cat(p, dim=-1)
            
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


