import os

from dataclasses import dataclass

from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import v2
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST

from sampler import MintermSampler

from typing import List, Tuple

@dataclass
class Batch:
    images: Tensor = None
    labels: Tensor = None

class CIFARTransform(object):
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
    
class MNISTTransform(object):
    def __init__(self, split: str) -> None:
        self.transform = T.Compose([T.ToTensor(),
                                    T.Normalize((0.1307,), (0.3081,))])
            
    def __call__(self, sample: Image) -> Tensor:
        x = self.transform(sample)
        return x
    
class FashionMNISTTransform(object):
    def __init__(self, split: str) -> None:
        self.transform = T.Compose([T.ToTensor(),
                                    T.Normalize((0.5,), (0.5,))])
            
    def __call__(self, sample: Image) -> Tensor:
        x = self.transform(sample)
        return x

class CELEBATransform(object):
    def __init__(self, split: str, img_size: int) -> None:
        if split.lower() == "train":
            self.transform = T.Compose([T.Resize(img_size),
                                        T.RandomHorizontalFlip(),
                                        T.ToTensor()])
        else:
            self.transform = T.Compose([T.Resize(img_size),
                                        T.ToTensor()])

    def __call__(self, sample: Image) -> Tensor:
        x = self.transform(sample)
        return x
    

class CXR14Transform(object):
    def __init__(self, split: str, img_size: int) -> None:
        if split.lower() == "train":
            self.transform = T.Compose([v2.ToImage(),
                                        v2.ToDtype(torch.uint8, scale=True),
                                        v2.Resize(img_size),
                                        v2.ToDtype(torch.float32, scale=True),
                                        v2.RandomHorizontalFlip()])
        else:
            self.transform = T.Compose([v2.ToImage(),
                                        v2.ToDtype(torch.uint8, scale=True),
                                        v2.Resize(img_size),
                                        v2.ToDtype(torch.float32, scale=True)])
            
    def __call__(self, sample: Image) -> Tensor:
        x = self.transform(sample)
        return x

class CXR14(object):
    def __init__(
        self,
        split: str,
        root: str,
        transform = None
    ):
        self.root = root
        self.transform = transform

        df = pd.read_csv(f"{root}/Data_Entry_2017_v2020.csv")

        self.im_filenames = df["Image Index"].to_list()
        self.labels = [labels.split("|") for labels in df["Finding Labels"].to_list()]

        self.classes = np.unique([c for label_list in self.labels for c in label_list])
        self.num_classes = len(self.classes)

        self.class_to_idx = {c : i for i, c in enumerate(self.classes)}

        self.im_filename_to_targets = {}
        for i in range(len(self.im_filenames)):
            im_filename = self.im_filenames[i]
            label_list = self.labels[i]
            target = torch.zeros((self.num_classes,))
            for label in label_list:
                idx = self.class_to_idx[label]
                target += F.one_hot(torch.tensor(idx), self.num_classes)
            self.im_filename_to_targets[im_filename] = target

        self.targets = torch.stack(list(self.im_filename_to_targets.values()), dim=0)
        self.targets = self.targets.to(torch.uint32).tolist()

        if split.lower() == "train":
            self.split_im_filenames = np.loadtxt(f"{root}/train_val_list.txt", dtype=np.chararray)
            self.split_im_filenames = self.split_im_filenames[:78544]
            self.targets = self.targets[:78544]
        elif split.lower() == "val":
            self.split_im_filenames = np.loadtxt(f"{root}/train_val_list.txt", dtype=np.chararray)
            self.split_im_filenames = self.split_im_filenames[78544:]
            self.targets = self.targets[78544:]
        elif split.lower() == "test":
            self.split_im_filenames = np.loadtxt(f"{root}/test_list.txt", dtype=np.chararray)
        else:
            raise ValueError(f"Unknown split {split}")
        
        self.length = len(self.split_im_filenames)

    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        im_filename = self.split_im_filenames[idx]
        im_path = f"{self.root}/images/{im_filename}"
        im = Image.open(im_path).convert('L')

        if self.transform:
            im = self.transform(im)
            im = im.repeat(3, 1, 1)
        return im, self.im_filename_to_targets[im_filename]



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
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        im_filename = self.im_filenames[idx]
        im_path = os.path.join(self.root, "img_align_celeba", im_filename)
        im = Image.open(im_path)
        im = im.crop((0, 40, im.size[0], im.size[1]))

        if self.transform:
            im = self.transform(im)

        return im, torch.tensor(self.targets[idx])

def collate_fn(batch) -> Batch:
    labels = torch.stack([b[1] for b in batch], dim=0)
    images = torch.stack([b[0] for b in batch], dim=0)
    return Batch(images=images, labels=labels)

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
    cifar100_root = "/mnt/localdisk/gabriel/nodocker/CIFAR100"
    celeb_a_root = "/mnt/localdisk/gabriel/nodocker/CELEBA"
    mnist_root = "/mnt/localdisk/gabriel/nodocker/MNIST"
    fashionmnist_root = "/mnt/localdisk/gabriel/nodocker/FashionMNIST"
    cxr14_root = "/mnt/localdisk/gabriel/nodocker/CXR14"

    print(dataset)

    if dataset == "MNIST":
        dataset = MNIST(
            root=mnist_root,
            train=(split == "train"),
            download=True,
            target_transform=lambda i : F.one_hot(torch.tensor(i), num_classes=10),
            transform=MNISTTransform(split))

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            collate_fn=collate_fn,
        )

    elif dataset == "FASHIONMNIST":
        dataset = FashionMNIST(
            root=fashionmnist_root,
            train=(split == "train"),
            download=True,
            target_transform=lambda i : F.one_hot(torch.tensor(i), num_classes=10),
            transform=FashionMNISTTransform(split)
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            collate_fn=collate_fn,
        )

    elif dataset == "CIFAR10":
        dataset = CIFAR10(
            root=cifar10_root,
            train=(split == "train"),
            download=True,
            target_transform=lambda i : F.one_hot(torch.tensor(i), num_classes=10),
            transform=CIFARTransform(split),
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            collate_fn=collate_fn,
        )

    elif dataset == "CIFAR100":
        dataset = CIFAR100(
            root=cifar100_root,
            train=(split == "train"),
            download=True,
            target_transform=lambda i : F.one_hot(torch.tensor(i), num_classes=100),
            transform=CIFARTransform(split),
        )

        if split == "train":
            dataloader = DataLoader(
                dataset,
                batch_sampler=MintermSampler(dataset.targets, batch_size, 100, 50),
                pin_memory=pin_memory,
                num_workers=num_workers,
                persistent_workers=persistent_workers,
                collate_fn=collate_fn,
            )
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=pin_memory,
                num_workers=num_workers,
                persistent_workers=persistent_workers,
                collate_fn=collate_fn,
            )

    elif dataset == "CELEBA":
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
                batch_sampler=MintermSampler(dataset.targets, batch_size, 5, 5),
                pin_memory=pin_memory,
                num_workers=num_workers,
                persistent_workers=persistent_workers,
                collate_fn=collate_fn,
            )
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=pin_memory,
                num_workers=num_workers,
                persistent_workers=persistent_workers,
                collate_fn=collate_fn,
            )
    elif dataset == "CXR14":
        dataset = CXR14(
            root=cxr14_root,
            split=split,
            transform=CXR14Transform(split=split, img_size=img_size),
        )

        if split == "train":
            dataloader = DataLoader(
                dataset,
                batch_sampler=MintermSampler(dataset.targets, batch_size, 15, 15),
                pin_memory=pin_memory,
                num_workers=num_workers,
                persistent_workers=persistent_workers,
                collate_fn=collate_fn,
            )
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=pin_memory,
                num_workers=num_workers,
                persistent_workers=persistent_workers,
                collate_fn=collate_fn,
            )
    else:
        raise ValueError(f"Unkown dataset {dataset}")
            
    return dataloader
