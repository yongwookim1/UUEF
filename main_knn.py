import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
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
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
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
    
    # ImageNet-1K
    (
        model,
        retain_loader,
        forget_loader,
        val_retain_loader,
        val_forget_loader
    ) = utils.setup_model_dataset(args)
    
    # Office-Home
    office_home_real_world_data_loader = utils.office_home_dataloaders(data_dir=args.office_home_dataset_path, domain="Real_World", batch_size=512, num_workers=4)
    office_home_art_data_loader = utils.office_home_dataloaders(data_dir=args.office_home_dataset_path, domain="Art", batch_size=512, num_workers=4)
    office_home_clipart_data_loader = utils.office_home_dataloaders(data_dir=args.office_home_dataset_path, domain="Clipart", batch_size=512, num_workers=4)
    office_home_product_data_loader = utils.office_home_dataloaders(data_dir=args.office_home_dataset_path, domain="Product", batch_size=512, num_workers=4)

    # CUB
    cub_data_loader = utils.cub_dataloaders(batch_size=512, data_dir=args.cub_dataset_path, num_workers=4)
    
    # DomainNet126
    domainnet126_clipart_data_loader = utils.domainnet126_dataloaders(batch_size=512, domain='clipart', data_dir=args.domainnet_dataset_path, num_workers=4)
    domainnet126_painting_data_loader = utils.domainnet126_dataloaders(batch_size=512, domain='painting', data_dir=args.domainnet_dataset_path, num_workers=4)
    domainnet126_real_data_loader = utils.domainnet126_dataloaders(batch_size=512, domain='real', data_dir=args.domainnet_dataset_path, num_workers=4)
    domainnet126_sketch_data_loader = utils.domainnet126_dataloaders(batch_size=512, domain='sketch', data_dir=args.domainnet_dataset_path, num_workers=4)
    
    dataset_names = ["imagenet_forget", "imagenet_retain","imagenet_val_forget", "imagenet_val_retain","office_home_real_world", "office_home_art", "office_home_clipart", "office_home_product", "cub", "domainnet126_clipart", "domainnet126_painting", "domainnet126_real", "domainnet126_sketch"]
    
    dataloaders = {}
    for i, dataloader in enumerate([forget_loader, retain_loader, val_forget_loader, val_retain_loader, office_home_real_world_data_loader, office_home_art_data_loader, office_home_clipart_data_loader, office_home_product_data_loader, cub_data_loader, domainnet126_clipart_data_loader, domainnet126_painting_data_loader, domainnet126_real_data_loader, domainnet126_sketch_data_loader]):
        train_size = int(0.8 * len(dataloader.dataset))
        test_size = len(dataloader.dataset) - train_size
        
        train_dataset, test_dataset = random_split(
            dataloader.dataset, [train_size, test_size]
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
        
        dataset_name = dataset_names[i]
        dataloaders[dataset_name] = (train_loader, test_loader)
    
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
    utils.setup_seed(2)
    
    args = arg_parser.parse_args()
    
    dataloaders = create_all_data_loaders(args)
    
    results = {}
    for name, (train_loader, test_loader) in dataloaders.items():
        print(f"Processing {name} loader")
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        
        model = utils.load_model(args.model_path, device)
        
        train_features, train_labels = extract_features(model, train_loader, device)
        test_features, test_labels = extract_features(model, test_loader, device)
        
        knn_accuracy = evaluate_knn(
            train_features,
            train_labels,
            test_features,
            test_labels,
            5,
        )
        results[name] = knn_accuracy

    for name, accuracy in results.items():
        print(f"{name}: {accuracy * 100:.2f}%")
            
    return results


if __name__ == "__main__":
    main()
