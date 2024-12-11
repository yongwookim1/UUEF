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
    def __init__(self, image_folder: str, transform: Optional[transforms.Compose] = None):
        self.image_folder = image_folder
        self.transform = transform
        self.images: List[str] = []
        self.labels: List[int] = []
        self.classes = sorted(os.listdir(image_folder))
        
        self._load_dataset()
    
    def _load_dataset(self) -> None:
        for label, cls in enumerate(self.classes):
            cls_folder = Path(self.image_folder) / cls
            if cls_folder.is_dir():
                for img_path in cls_folder.glob("*.[jp][pn][g]"):
                    self.images.append(str(img_path))
                    self.labels.append(label)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        return image, label


def load_model(model_path: str, device) -> torch.nn.Module:
    model = models.resnet50(weights=None)
    checkpoint = torch.load(model_path, map_location=device)
    
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
        
    state_dict = {
        k.replace("module.", ""): v for k, v in checkpoint.items()
        if not k.startswith('normalize.')
    }
    
    model.load_state_dict(state_dict, strict=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    return model.to(device)


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
    
    dataset = OfficeHomeDataset(args.office_home_dataset_path, transform=transform)
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

def evaluate_office_home_knn(model):
    utils.setup_seed(2)
    
    args = arg_parser.parse_args()
    
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
    
    train_loader, test_loader = create_data_loaders(args)
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    model_paths = [
        args.retrained_model_path
    ]
    model = load_model(model_paths[0], device)
    
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


if __name__ == "__main__":
    main()