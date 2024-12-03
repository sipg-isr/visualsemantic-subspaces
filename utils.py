import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from tqdm import tqdm

def show_dataset(dataloader, n, figsize=10):
    plt.figure(figsize=(figsize,figsize))

    for i in range(n**2):
        plt.subplot(n,n,i+1)
        data = dataloader.dataset[np.random.randint(len(dataloader.dataset))]
        im = data[0].permute(1,2,0)
        im = (im - im.min()) / (im.max() - im.min())
        plt.imshow(im)
        plt.axis('off')


def embed_dataset(model,
                  dataset,
                  normalize=False,
                  pin_memory=True,
                  num_workers=4,
                  transform=None,
                  device='cuda'):
    model.eval()

    if transform is not None:
        dataset.transform = transform
        
    dataloader = DataLoader(dataset,
                            batch_size=512,
                            shuffle=False,
                            pin_memory=pin_memory,
                            num_workers=num_workers)
    
    x, y = [], []
    for data in tqdm(dataloader):
        ims, targets = data
        ims = ims.to(device)
        with torch.no_grad():
            embeddings = model(ims).cpu().detach()
        x.append(embeddings)
        y.append(targets)

    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)

    if normalize:
        x = F.normalize(x, dim=-1, p=2)

    return x, y


def default_corrupt(trainset, ratio):
    train_labels = np.asarray(trainset.targets)
    num_classes = np.max(train_labels) + 1
    n_train = len(train_labels)
    n_rand = int(len(trainset.data)*ratio)
    randomize_indices = np.random.choice(range(n_train), size=n_rand, replace=False)
    train_labels[randomize_indices] = np.random.choice(np.arange(num_classes), size=n_rand, replace=True)
    trainset.targets = torch.tensor(train_labels).int().tolist()
    return trainset