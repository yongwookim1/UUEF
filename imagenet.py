import os
import sys

import torch
import torchvision
from torchvision import transforms, datasets
from datasets.load import load_dataset
from torch.utils.data import DataLoader, Subset

# sys.path.append(".")
# from cfg import *
from tqdm import tqdm


def prepare_data(
    dataset,
    batch_size=512,
    shuffle=True,
    train_subset_indices=None,
    val_subset_indices=None,
    data_path="/home/dataset/imagenet1k/data",
    args=None,
):
    if dataset == "imagenet":
        train_dir = os.path.join(data_path, "train")
        val_dir = os.path.join(data_path, "val")

        train_transform = transforms.Compose([
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.RandomResizedCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        
        if args.original_Df:
            train_transform = transforms.Compose([
                transforms.Lambda(lambda x: x.convert("RGB")),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

        val_transform = transforms.Compose([
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
        ])

        train_set = datasets.ImageFolder(train_dir, transform=train_transform)
        validation_set = datasets.ImageFolder(val_dir, transform=val_transform)
    else:
        raise NotImplementedError

    if train_subset_indices is not None:
        forget_indices = torch.ones_like(train_subset_indices) - train_subset_indices
        train_subset_indices = torch.nonzero(train_subset_indices)

        forget_indices = torch.nonzero(forget_indices)
        retain_set = Subset(train_set, train_subset_indices)
        forget_set = Subset(train_set, forget_indices)
        
    if val_subset_indices is not None:
        val_subset_indices = torch.nonzero(val_subset_indices).squeeze()
        validation_set = Subset(validation_set, val_subset_indices)

    if train_subset_indices is not None:
        loaders = {
            "train": DataLoader(retain_set, batch_size=batch_size, num_workers=8, shuffle=shuffle),
            "val": DataLoader(validation_set, batch_size=batch_size, num_workers=8, shuffle=shuffle),
            "fog": DataLoader(forget_set, batch_size=batch_size, num_workers=8, shuffle=shuffle),
        }
    else:
        loaders = {
            "train": DataLoader(train_set, batch_size=batch_size, num_workers=8, shuffle=shuffle),
            "val": DataLoader(validation_set, batch_size=batch_size, num_workers=8, shuffle=shuffle),
        }
    return loaders


def get_x_y_from_data_dict(data, device):
    x, y = data
    x, y = x.to(device), y.to(device)
    return x, y


if __name__ == "__main__":
    ys = {}
    ys["train"] = []
    ys["val"] = []
    loaders = prepare_data(
        dataset="imagenet", batch_size=1, shuffle=False, data_path="/home/dataset/imagenet1k/data"
    )
    for data in tqdm(loaders["val"], ncols=100):
        x, y = get_x_y_from_data_dict(data, "cpu")
        ys["val"].append(y.item())
    for data in tqdm(loaders["train"], ncols=100):
        x, y = get_x_y_from_data_dict(data, "cpu")
        ys["train"].append(y.item())
    ys["train"] = torch.Tensor(ys["train"]).long()
    ys["val"] = torch.Tensor(ys["val"]).long()
    torch.save(ys["train"], "train_ys.pth")
    torch.save(ys["val"], "val_ys.pth")
