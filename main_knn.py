import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms, datasets
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

import utils
import arg_parser


class OfficeHomeDataset(Dataset):
    def __init__(self, image_folder, domain, transform=None):
        self.image_folder = image_folder + "/" + domain
        self.images = []
        self.labels = []
        self.transform = transform

        self.classes = sorted(os.listdir(self.image_folder))
        for label, cls in enumerate(self.classes):
            cls_folder = os.path.join(self.image_folder, cls)
            if os.path.isdir(cls_folder):
                for img_name in os.listdir(cls_folder):
                    img_path = os.path.join(cls_folder, img_name)
                    if img_path.endswith('.jpg') or img_path.endswith('.png'):
                        self.images.append(img_path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    
    
class DomainNet126(Dataset):
    def __init__(self, root, domain, train=True, transform=None, from_file=False):
        
        if not from_file:
            data = []
            labels = []

            f = open(os.path.join(root,domain+"_list.txt"), "r")
            lines = f.readlines()
            lines = [l.split(" ") for l in lines]
            lines = np.array(lines)

            files = lines[:-1,0]
            files = [os.path.join(root, sfile) for sfile in files]

            classes = lines[:-1,1]
            classes = [int(c[:-1]) for c in classes]

            data.extend(files)
            labels.extend(classes)

            self.data = np.array(data)
            self.labels = np.array(labels)
            self.transform = transform

        else:
            data = np.load(os.path.join(root, domain+"_imgs.npy"))
            labels = np.load(os.path.join(root, domain+"_labels.npy"))
        
            np.random.seed(42)
            idx = np.random.permutation(len(data))

            self.data = np.array(data)[idx]
            self.labels = np.array(labels)[idx]

            test_perc = 20
            
            test_len = len(self.data)*test_perc//100                   
            
            if train:
                self.data = self.data[test_len:]
                self.labels = self.labels[test_len:]
            else:
                self.data = self.data[:test_len]
                self.labels = self.labels[:test_len]


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]          

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(img)
        
        if self.transform is not None:
            img = self.transform(img)

        return img, target


@torch.no_grad()
def extract_features(model, loader: DataLoader, device) -> Tuple[np.ndarray, np.ndarray]:
    features, labels = [], []
    model.eval()
    
    for x, y in tqdm(loader):
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        features.append(outputs.cpu())
        labels.append(y.cpu())
    
    features_tensor = torch.cat(features).squeeze().cpu().numpy()
    labels_array = torch.cat(labels).cpu().numpy()
    return features_tensor, labels_array


def create_data_loaders(args):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = OfficeHomeDataset(args.office_home_dataset_path, domain="Real_World", transform=transform)
    
    train_size = int(0.8 * len(dataloader.dataset))
    test_size = len(dataloader.dataset) - train_size
    
    g = torch.Generator()
    g.manual_seed(2)
    train_dataset, test_dataset = random_split(
        dataloader.dataset, [train_size, test_size], generator=g
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=4,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=4,
    )
    
    return train_loader, test_loader


def create_all_data_loaders(args):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataloaders = {}
    
    office_home_domains = ["Real_World", "Art", "Clipart", "Product"]
    for domain in office_home_domains:
        
        dataset = OfficeHomeDataset(args.office_home_dataset_path, domain=domain, transform=transform)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        
        g = torch.Generator()
        g.manual_seed(2)
        train_dataset, test_dataset = random_split(
            dataset, [train_size, test_size], generator=g
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=512,
            shuffle=False,
            num_workers=4,
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=512,
            shuffle=False,
            num_workers=4,
        )
        
        dataloaders[f"office_home_{domain.lower()}"] = (train_loader, test_loader)
    
    # CUB
    cub_dataset = datasets.ImageFolder(root=args.cub_dataset_path, transform=transform)
    
    train_size = int(0.8 * len(cub_dataset))
    test_size = len(cub_dataset) - train_size
    
    g = torch.Generator()
    g.manual_seed(2)
    train_dataset, test_dataset = random_split(
        cub_dataset, [train_size, test_size], generator=g
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=4,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=4,
    )
    
    dataloaders["cub"] = (train_loader, test_loader)
    
    # DomainNet126
    domainnet_domains = ['clipart', 'painting', 'real', 'sketch']
    for domain in domainnet_domains:
        dataset = DomainNet126(args.domainnet_dataset_path, domain=domain, transform=transform)
        
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        
        g = torch.Generator()
        g.manual_seed(2)
        train_dataset, test_dataset = random_split(
            dataset, [train_size, test_size], generator=g
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=512,
            shuffle=False,
            num_workers=4,
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=512,
            shuffle=False,
            num_workers=4,
        )
        
        dataloaders[f"domainnet126_{domain}"] = (train_loader, test_loader)
    
    return dataloaders


def evaluate_knn(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    k: int
) -> float:
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    knn.fit(train_features, train_labels)
    return knn.score(test_features, test_labels)


def evaluate_office_home_knn(model, args):
    utils.setup_seed(2)
    
    train_loader, test_loader = create_data_loaders(args)
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    train_features, train_labels = extract_features(model, train_loader, device)
    test_features, test_labels = extract_features(model, test_loader, device)
    
    knn_accuracy = evaluate_knn(
        train_features,
        train_labels,
        test_features,
        test_labels,
        5,
    )
    print(f"kNN(k=5) accuracy: {knn_accuracy * 100:.2f}%")
    return knn_accuracy


def main():
    args = arg_parser.parse_args()
    utils.setup_seed(2)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    dataloaders = create_all_data_loaders(args)
    
    results = {}
    model = utils.load_model(args.model_path, arch=args.arch, device=device)
    
    for name, (train_loader, test_loader) in dataloaders.items():
        print(f"Processing {name} loader")
        
        train_features, train_labels = extract_features(model, train_loader, device)
        test_features, test_labels = extract_features(model, test_loader, device)
        
        knn_accuracy = evaluate_knn(
            train_features,
            train_labels,
            test_features,
            test_labels,
            5,
        )
        print(f"kNN(k=5) accuracy: {knn_accuracy * 100:.2f}%")

    for name, accuracy in results.items():
        print(f"{name}: {accuracy * 100:.2f}%")
            
    return results


if __name__ == "__main__":
    main()
