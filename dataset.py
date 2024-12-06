import os

from dataclasses import dataclass

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torch.utils.data.sampler import WeightedRandomSampler

from typing import List

@dataclass
class Batch:
    images: Tensor = None
    labels: Tensor = None


class CIFAR10Transform(object):
    def __init__(self, split: str) -> None:
        mu  = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

        if split.lower() == "train":
            self.transform = T.Compose([T.RandomCrop(32, padding=4),
                                        T.RandomHorizontalFlip(),
                                        T.ToTensor(),
                                        T.Normalize(mu,std)])
        else:
            self.transform = T.Compose([T.ToTensor(),
                                        T.Normalize(mu,std)])
            
    def __call__(self, sample: Image) -> Tensor:
        x = self.transform(sample)
        return x

class CELEBATransform(object):
    def __init__(self, split: str, img_size: int) -> None:
        if split.lower() == "train":
            self.transform = T.Compose([T.Resize(img_size),
                                        T.RandomHorizontalFlip(),
                                        T.ColorJitter(0.2,0.2,0.2,0.1),
                                        T.RandomGrayscale(p=0.5),
                                        T.ToTensor()])
        else:
            self.transform = T.Compose([T.Resize(img_size),
                                        T.ToTensor()])

    def __call__(self, sample: Image) -> Tensor:
        x = self.transform(sample)
        return x
    
class CELEBA(Dataset):
    def __init__(
        self,
        root : str,
        split : str,
        class_select=None,
        transform=None
    ) -> None:
        super(CELEBA, self).__init__()

        self.root = root
        self.split = split.lower()
        self.transform = transform

        self.splits = {
            "train" : 0,
            "val"   : 1,
            "test"  : 2,
        }

        split_im_filenames = set()
        with open(os.path.join(self.root, "list_eval_partition.txt")) as f:
            for line in f:
                im_filename, split = line.split()
                if int(split) == self.splits[self.split]:
                    split_im_filenames.add(im_filename)

        self.im_filenames = []
        self.targets = []

        with open(os.path.join(self.root, "list_attr_celeba.txt")) as f:
            for i, line in enumerate(f):
                data = line.rstrip().split()
                if i == 1:
                    self.classes = data
                    self.class_to_idx = {c : i for i, c in enumerate(self.classes)}
                    if class_select is not None:
                        self.classes = class_select
                if i > 1:
                    im_filename = data[0]
                    if im_filename in split_im_filenames:
                        targets = [max(int(data[1+self.class_to_idx[c]]), 0) for c in self.classes]
                        if sum(targets) > 0:
                            self.im_filenames.append(im_filename)
                            self.targets.append(targets)

        self.class_to_idx = {t : i for i, t in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.im_filenames)
    
    def __getitem__(self, idx: int) -> Batch:
        im_filename = self.im_filenames[idx]
        im_path = os.path.join(self.root, "img_align_celeba", im_filename)
        im = Image.open(im_path)
        im = im.crop((0, 40, im.size[0], im.size[1]))

        if self.transform:
            im = self.transform(im)

        return Batch(images=im, labels=torch.tensor(self.targets[idx]))
    
    def collate_fn(self, batch: List[Batch]) -> Batch:
        labels = torch.stack([b.labels for b in batch], dim=0)
        images = torch.stack([b.images for b in batch], dim=0)
        return Batch(images=images, labels=labels)


def get_weighted_sampler(targets : List[List]):
    t = torch.tensor(targets)
    cnf, counts = torch.unique(t, dim=0, return_counts=True)
    weights = torch.zeros((len(targets),))

    for i, clause in enumerate(cnf):
        mask = (t == clause).all(dim=-1)
        weights[mask] = 1.0 / counts[i]

    sampler = WeightedRandomSampler(weights, len(weights)) 
    return sampler


class MintermSampler():
    def __init__(self, targets: List[List[int]], batch_size: int) -> None:
        self._batch_size = batch_size
        self._n_samples = len(targets)
        self._n_literals = len(targets[0])
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

    def __iter__(self):        
        for _ in range(self._n_batches):
            sel = np.random.choice(
                np.arange(self._n_minterms),
                self._n_literals,
                replace=False
            )
            batch = []
            for minterm_label in sel:
                batch.append(np.random.choice(self._minterm2idx[minterm_label],
                                              self._batch_size // self._n_literals))
            batch = np.concatenate(batch)
            yield (batch)

    def __len__(self) -> int:
        return self._n_batches


def get_loader(
    dataset,
    split: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
) -> DataLoader:
    split = split.lower()
    dataset = dataset.upper()

    cifar10_root = "/mnt/localdisk/gabriel/nodocker/CIFAR10"
    celeb_a_root = "/mnt/localdisk/gabriel/nodocker/CELEBA"

    if dataset == "CIFAR10":
        dataset = CIFAR10(
            root=cifar10_root,
            train=(split == "train"),
            download=True,
            target_transform=lambda idx : F.one_hot(torch.tensor(idx), num_classes=10),
            transform=CIFAR10Transform(split),
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )
            
    if dataset == "CELEBA":
        dataset = CELEBA(
            root=celeb_a_root,
            class_select=["Bald",
                          "Eyeglasses",
                          "Wearing_Necktie",
                          "Wearing_Hat",
                          "Male"],
            split=split, 
            transform=CELEBATransform(split=split, img_size=img_size),
        )

        if split == "train":
            dataloader = DataLoader(
                dataset,
                #batch_size=batch_size,
                #sampler=get_weighted_sampler(dataset.targets),
                batch_sampler=MintermSampler(dataset.targets, batch_size),
                pin_memory=pin_memory,
                num_workers=num_workers,
                persistent_workers=persistent_workers,
                collate_fn=dataset.collate_fn,
            )
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=pin_memory,
                num_workers=num_workers,
                persistent_workers=persistent_workers,
                collate_fn=dataset.collate_fn,
            )   
            
    return dataloader
