## [Learning Visual-Semantic Subspace Representations](https://openreview.net/forum?id=R3O1mD9lyZ)

Gabriel Moreira, Manuel Marques, Joao Costeira, Alexander G Hauptmann

Published 2025AISTATS 2025

[PDF](https://raw.githubusercontent.com/mlresearch/v258/main/assets/moreira25a/moreira25a.pdf)

Abstract:

Learning image representations that capture rich semantic relationships remains a significant challenge. Existing approaches are either contrastive, lacking robust theoretical guarantees, or struggle to effectively represent the partial orders inherent to structured visual-semantic data. In this paper, we introduce a nuclear norm-based loss function, grounded in the same information theoretic principles that have proved effective in self-supervised learning. We present a theoretical characterization of this loss, demonstrating that, in addition to promoting class orthogonality, it encodes the spectral geometry of the data within a subspace lattice. This geometric representation allows us to associate logical propositions with subspaces, ensuring that our learned representations adhere to a predefined symbolic structure.


![image](https://github.com/user-attachments/assets/21c3ef51-765f-49c1-ad04-b591f3660788)

The capability of explicitely represent negations is one key feature of this subspace representation. The figure shows that massive systems like CLIP that do not have structured representations by design do not cope with "negations" among other queries whereas by imposing a subspace structure the negation is naturally represented by the orthogonal complement.

## Code

# Run training

> python ./train.py --config-name=celeb-a general.name="name_of_experiment"
