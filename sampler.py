import numpy as np
from typing import List, Optional

class MintermSampler():
    def __init__(
        self,
        targets: List[List[int]],
        batch_size: int,
        n_literals: int,
        n_minterms_per_batch: Optional[int] = None
    ) -> None:
        self._batch_size = batch_size
        self._n_samples = len(targets)
        self._n_literals = n_literals
        self._targets = np.array(targets)

        self._n_batches = self._n_samples // self._batch_size
        self._minterms, self._minterm_labels = np.unique(
            self._targets,
            axis=0,
            return_inverse=True,
        )

        self._n_minterms = self._minterm_labels.max() + 1

        self._minterm2idx = []
        for m in range(self._n_minterms):
            self._minterm2idx.append(np.where(self._minterm_labels == m)[0])

        if n_minterms_per_batch is None:
            self._n_minterms_per_batch = self._n_literals
        else:
            self._n_minterms_per_batch = n_minterms_per_batch

    def __iter__(self):        
        for _ in range(self._n_batches):
            sel = np.random.choice(
                np.arange(self._n_minterms),
                self._n_minterms_per_batch,
                replace=False
            )
            batch = []
            for minterm_label in sel:
                batch.append(np.random.choice(self._minterm2idx[minterm_label],
                                              self._batch_size // self._n_minterms_per_batch))
            batch = np.concatenate(batch)
            yield (batch)

    def __len__(self) -> int:
        return self._n_batches