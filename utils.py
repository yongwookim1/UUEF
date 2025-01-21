"""
    setup model and datasets
"""


import copy
import os
import random
import shutil
import sys
import time
from typing import Tuple

import numpy as np
import torch
import wandb
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
from tqdm import tqdm

from CKA.CKA import CudaCKA
from dataset import *
from dataset import TinyImageNet
from imagenet import prepare_data
from models import *
from models.ConvNeXt import convnext_tiny


__all__ = [
    "setup_model_dataset",
    "AverageMeter",
    "warmup_lr",
    "save_checkpoint",
    "setup_seed",
    "accuracy",
]


def init_wandb(args, project_name="unlearning"):
    """initialize wandb configuration"""
    run_name = f"{args.wandb_name}"
    
    # initialize wandb
    wandb.init(
        project=project_name,
        name=run_name,
        config=vars(args),
        dir=os.path.join(args.save_dir),
    )
    
    return wandb.run


def create_data_loaders(args):
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
    ) = setup_model_dataset(args)
    
    # Office-Home
    office_home_real_world_data_loader = office_home_dataloaders(data_dir=args.office_home_dataset_path, domain="Real_World", batch_size=512, num_workers=4)
    office_home_art_data_loader = office_home_dataloaders(data_dir=args.office_home_dataset_path, domain="Art", batch_size=512, num_workers=4)
    office_home_clipart_data_loader = office_home_dataloaders(data_dir=args.office_home_dataset_path, domain="Clipart", batch_size=512, num_workers=4)
    office_home_product_data_loader = office_home_dataloaders(data_dir=args.office_home_dataset_path, domain="Product", batch_size=512, num_workers=4)

    # CUB
    cub_data_loader = cub_dataloaders(batch_size=512, data_dir=args.cub_dataset_path, num_workers=4)
    
    # DomainNet126
    domainnet126_clipart_data_loader = domainnet126_dataloaders(batch_size=512, domain='clipart', data_dir=args.domainnet_dataset_path, num_workers=4)
    domainnet126_painting_data_loader = domainnet126_dataloaders(batch_size=512, domain='painting', data_dir=args.domainnet_dataset_path, num_workers=4)
    domainnet126_real_data_loader = domainnet126_dataloaders(batch_size=512, domain='real', data_dir=args.domainnet_dataset_path, num_workers=4)
    domainnet126_sketch_data_loader = domainnet126_dataloaders(batch_size=512, domain='sketch', data_dir=args.domainnet_dataset_path, num_workers=4)
    
    dataset_names = ["imagenet_forget", "imagenet_retain", "imagenet_val_forget", "imagenet_val_retain", "office_home_real_world", "office_home_art", "office_home_clipart", "office_home_product", "cub", "domainnet126_clipart", "domainnet126_painting", "domainnet126_real", "domainnet126_sketch"]
    
    data_loaders = {}
    for i, dataloader in enumerate([forget_loader, retain_loader, val_forget_loader, val_retain_loader, office_home_real_world_data_loader, office_home_art_data_loader, office_home_clipart_data_loader, office_home_product_data_loader, cub_data_loader, domainnet126_clipart_data_loader, domainnet126_painting_data_loader, domainnet126_real_data_loader, domainnet126_sketch_data_loader]):
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
        
        dataset_name = dataset_names[i]
        data_loaders[dataset_name] = (train_loader, test_loader)
    
    return data_loaders


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


def office_home_real_world_knn(model, args):
    setup_seed(2)
    data_loaders = create_data_loaders(args)
    train_loader, test_loader = data_loaders["office_home_real_world"]
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    train_features, train_labels = extract_features(model, train_loader, device)
    test_features, test_labels = extract_features(model, test_loader, device)
    
    knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
    knn.fit(train_features, train_labels)
    score = knn.score(test_features, test_labels)
    score = float(f"{score * 100:.2f}")
    
    return {"office_home_real_world": score}


def evaluate_knn(model_path, args):
    setup_seed(2)
    data_loaders = create_data_loaders(args)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    knn_accuracy = {}
    for dataset_name, (train_loader, test_loader) in data_loaders.items():
        if "imagenet" in dataset_name:
            model = initialize_model(model_path, device, arch=args.arch)
        else:
            model = load_model(model_path, device, args.arch)
        
        if dataset_name == "imagenet_retain":
            torch.manual_seed(2) # fix random seed for reproducibility
            dataset_size = len(train_loader.dataset)
            subset_size = int(0.1 * dataset_size)
            indices = torch.randperm(dataset_size)[:subset_size]
            train_subset = torch.utils.data.Subset(train_loader.dataset, indices)
            train_loader = torch.utils.data.DataLoader(
                train_subset,
                batch_size=train_loader.batch_size,
                shuffle=False,
                num_workers=train_loader.num_workers
            )
        
        train_features, train_labels = extract_features(model, train_loader, device)
        test_features, test_labels = extract_features(model, test_loader, device)
        
        knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
        knn.fit(train_features, train_labels)
        score = knn.score(test_features, test_labels)
        score = float(f"{score * 100:.2f}")
        knn_accuracy[dataset_name] = score
        
    return knn_accuracy


class FeatureExtractor:
    def __init__(self):
        self.features = []
        
    def hook_fn(self, module, input, output):
        self.features.append(output)
        
    def clear(self):
        self.features = []


def evaluate_cka(unlearned_model, retrained_model, data_loader, device, args, mode='avgpool', Dr_features=None, Df_features=None, data=None):
    """
    compute CKA similarity between two models with feature reuse
    """
    # initialize CKA computation
    
    data_loader = DataLoader(
        data_loader.dataset,
        batch_size=data_loader.batch_size,
        shuffle=False,  # shuffle=False
        num_workers=data_loader.num_workers
    )
    
    cuda_cka = CudaCKA(device)
    layers = ['layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc'] if mode == 'all' else ['avgpool']
    cka_results = {layer: 0 for layer in layers}

    # if we have pre-computed features for both models, use them directly
    if (Dr_features is not None or Df_features is not None) and os.path.exists(os.path.join('features', f'{data}_{args.class_to_replace}_retrained_features.pt')):
        saved_features = torch.load(os.path.join('features', f'{data}_{args.class_to_replace}_retrained_features.pt'), map_location="cpu")
        
        for batch_idx in tqdm(range(len(data_loader))):
            for i, layer in enumerate(layers):
                # get unlearned features
                f_u = (Dr_features[batch_idx][i] if Dr_features is not None else Df_features[batch_idx][i]).to(device)
                f_u = f_u.view(f_u.size(0), -1)

                # get retrained features
                f_r = saved_features[batch_idx][i].to(device)
                f_r = f_r.view(f_r.size(0), -1)

                # calculate CKA
                cka_results[layer] += cuda_cka.linear_CKA(f_u, f_r).cpu()

                # clear GPU memory
                del f_u, f_r
                # torch.cuda.empty_cache()

        # average results
        n = len(data_loader)
        layer_results = {layer: {'cka': float(f"{(value / n).cpu().numpy() * 100:.2f}")} for layer, value in cka_results.items()}
        return {'cka': float(f"{sum(result['cka'] for result in layer_results.values()) / len(layer_results):.2f}")}

    elif (Dr_features is not None or Df_features is not None):
        features_dir = os.path.join('features')
        os.makedirs(features_dir, exist_ok=True)
        
        # define paths for feature files
        features_path = os.path.join(features_dir, f'{data}_{args.class_to_replace}_retrained_features.pt') if data else None
        
        retrained_model.eval()
        retrained_features = []
        retrained_extractor = FeatureExtractor()
        
        for batch_idx, (img, _) in enumerate(tqdm(data_loader)):
            img = img.to(device)
            
            # register hooks
            hooks = [getattr(retrained_model, layer).register_forward_hook(retrained_extractor.hook_fn) for layer in layers]

            # extract features
            with torch.no_grad():
                retrained_model(img)

            batch_features = []
            for i, layer in enumerate(layers):
                # get unlearned features
                if Dr_features is not None:
                    f_u = Dr_features[batch_idx][i].to(device)
                elif Df_features is not None:
                    f_u = Df_features[batch_idx][i].to(device)
                f_u = f_u.view(f_u.size(0), -1)

                # get retrained features
                f_r = retrained_extractor.features[i].to(device)
                batch_features.append(f_r.cpu())
                f_r = f_r.view(f_r.size(0), -1)

                # calculate CKA
                cka_results[layer] += cuda_cka.linear_CKA(f_u, f_r).cpu()

                # clear GPU memory
                del f_u, f_r
                # torch.cuda.empty_cache()

            retrained_features.append(batch_features)

            # cleanup
            for hook in hooks:
                hook.remove()
            retrained_extractor.clear()

        # save features if path is provided
        if features_path:
            torch.save(retrained_features, features_path)
        
        # average results
        n = len(data_loader)
        layer_results = {layer: {'cka': float(f"{(value / n).cpu().numpy() * 100:.2f}")} for layer, value in cka_results.items()}
        return {'cka': float(f"{sum(result['cka'] for result in layer_results.values()) / len(layer_results):.2f}")}

    elif os.path.exists(os.path.join('features', f'{data}_{args.class_to_replace}_retrained_features.pt')):
        features_dir = os.path.join('features')
        saved_features = torch.load(os.path.join(features_dir, f'{data}_{args.class_to_replace}_retrained_features.pt'), map_location="cpu")
        
        unlearned_model.eval()

        print(f"Extracting saved features...")
        unlearned_extractor = FeatureExtractor()
        
        # compute CKA
        for batch_idx, (img, _) in enumerate(tqdm(data_loader)):
            img = img.to(device)
            
            # register hooks
            hooks = [getattr(unlearned_model, layer).register_forward_hook(unlearned_extractor.hook_fn) for layer in layers]

            # extract features
            with torch.no_grad():
                unlearned_model(img)

            # compute CKA for each layer
            for i, layer in enumerate(layers):
                # get unlearned features
                f_u = unlearned_extractor.features[i].to(device)
                f_u = f_u.view(f_u.size(0), -1)

                # get retrained features
                f_r = saved_features[batch_idx][i].to(device)
                f_r = f_r.view(f_r.size(0), -1)

                # calculate CKA
                cka_results[layer] += cuda_cka.linear_CKA(f_u, f_r).cpu()
                
                # clear GPU memory
                del f_u, f_r
                # torch.cuda.empty_cache()

            # cleanup
            for hook in hooks:
                hook.remove()
            unlearned_extractor.clear()

        # average results
        n = len(data_loader)
        layer_results = {layer: {'cka': float(f"{(value / n).cpu().numpy() * 100:.2f}")} for layer, value in cka_results.items()}
        return {'cka': float(f"{sum(result['cka'] for result in layer_results.values()) / len(layer_results):.2f}")}
    else:
        os.makedirs('features', exist_ok=True)
        features_dir = os.path.join('features')
        features_path = os.path.join(features_dir, f'{data}_{args.class_to_replace}_retrained_features.pt') if data else None
        
        unlearned_model.eval()
        retrained_model.eval()
        
        unlearned_extractor = FeatureExtractor()
        retrained_extractor = FeatureExtractor()
        retrained_features = []
        
        for batch_idx, (img, _) in enumerate(tqdm(data_loader)):
            img = img.to(device)
            
            hooks = [
                getattr(unlearned_model, layer).register_forward_hook(unlearned_extractor.hook_fn)
                for layer in layers
            ] + [
                getattr(retrained_model, layer).register_forward_hook(retrained_extractor.hook_fn)
                for layer in layers
            ]
            
            with torch.no_grad():
                unlearned_model(img)
                retrained_model(img)
            
            batch_features = []
            for i, layer in enumerate(layers):
                f_u = unlearned_extractor.features[i].to(device)
                f_u = f_u.view(f_u.size(0), -1)
                f_r = retrained_extractor.features[i].to(device)
                batch_features.append(f_r.cpu())
                f_r = f_r.view(f_r.size(0), -1)
                
                cka_results[layer] += cuda_cka.linear_CKA(f_u, f_r).cpu()
                
                del f_u, f_r
                # torch.cuda.empty_cache()
            
            retrained_features.append(batch_features)
            
            for hook in hooks:
                hook.remove()
            unlearned_extractor.clear()
            retrained_extractor.clear()
            
        if features_path:
            torch.save(retrained_features, features_path)

        n = len(data_loader)
        layer_results = {layer: {'cka': float(f"{(value / n).cpu().numpy() * 100:.2f}")} for layer, value in cka_results.items()}
        return {'cka': float(f"{sum(result['cka'] for result in layer_results.values()) / len(layer_results):.2f}")}
            

def load_model(pretrained_model_path, device, arch):
    print(f"\nLoading model: {pretrained_model_path}")
    if arch == "resnet50":
        model = models.resnet50(weights=None)
    elif arch == "convnext_tiny":
        model = convnext_tiny(pretrained=False, normalize_layer=False)
    checkpoint = torch.load(pretrained_model_path, map_location=device)
    if "state_dict" in checkpoint.keys():
        checkpoint = checkpoint["state_dict"]
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    state_dict = {k: v for k, v in state_dict.items() if not (k.startswith('normalize.'))}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    print(f"Model loading complete: {pretrained_model_path}")
    return model


def initialize_model(model_path, device, imagenet=True, arch="resnet50"):
    """initialize and load a ResNet50 model from checkpoint
    
    Args:
        model_path (str): Path to model checkpoint
        device (torch.device): Device to load model to
        imagenet (bool): Whether to use ImageNet pretrained weights
        
    Returns:
        model (nn.Module): Initialized and loaded model
    """
    if arch == "resnet50":
        model = resnet50(imagenet=imagenet)
    elif arch == "convnext_tiny":
        model = convnext_tiny(imagenet=imagenet, normalize_layer=True)
    
    # Add normalization layer
    normalization = NormalizeByChannelMeanStd(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    model.normalize = normalization

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if "state_dict" in checkpoint.keys():
        checkpoint = checkpoint["state_dict"]
    checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint, strict=True)
    model = model.to(device)
    model.eval()
    
    return model


def warmup_lr(epoch, step, optimizer, one_epoch_step, args):
    overall_steps = args.warmup * one_epoch_step
    current_steps = epoch * one_epoch_step + step

    lr = args.lr * current_steps / overall_steps
    lr = min(lr, args.lr)

    for p in optimizer.param_groups:
        p["lr"] = lr


def save_checkpoint(
    state, is_SA_best, save_path, pruning, filename="checkpoint.pth.tar"
):
    filepath = os.path.join(save_path, str(pruning) + filename)
    torch.save(state, filepath)
    if is_SA_best:
        shutil.copyfile(
            filepath, os.path.join(save_path, str(pruning) + "model_SA_best.pth.tar")
        )


def load_checkpoint(device, save_path, pruning, filename="checkpoint.pth.tar"):
    filepath = os.path.join(save_path, str(pruning) + filename)
    if os.path.exists(filepath):
        print("Load checkpoint from:{}".format(filepath))
        return torch.load(filepath, device)
    print("Checkpoint not found! path:{}".format(filepath))
    return None


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def dataset_convert_to_train(dataset):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    dataset.transform = train_transform
    dataset.train = False


def dataset_convert_to_test(dataset, args=None):
    if args.dataset == "TinyImagenet":
        test_transform = transforms.Compose([])
    elif args.dataset == "imagenet":
        test_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )
    else:
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    dataset.transform = test_transform
    dataset.train = False


def setup_model_dataset(args):
    if args.dataset == "cifar10":
        classes = 10
        normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
        )
        train_full_loader, val_loader, _ = cifar10_dataloaders(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers
        )
        marked_loader, _, test_loader = cifar10_dataloaders(
            batch_size=args.batch_size,
            data_dir=args.data,
            num_workers=args.workers,
            class_to_replace=args.class_to_replace,
            num_indexes_to_replace=args.num_indexes_to_replace,
            indexes_to_replace=args.indexes_to_replace,
            seed=args.seed,
            only_mark=True,
            shuffle=True,
            no_aug=args.no_aug,
        )

        if args.train_seed is None:
            args.train_seed = args.seed
        setup_seed(args.train_seed)

        if args.imagenet_arch:
            model = model_dict[args.arch](num_classes=classes, imagenet=True)
        else:
            model = model_dict[args.arch](num_classes=classes)

        setup_seed(args.train_seed)

        model.normalize = normalization
        return model, train_full_loader, val_loader, test_loader, marked_loader
    elif args.dataset == "svhn":
        classes = 10
        normalization = NormalizeByChannelMeanStd(
            mean=[0.4377, 0.4438, 0.4728], std=[0.1201, 0.1231, 0.1052]
        )
        train_full_loader, val_loader, _ = svhn_dataloaders(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers
        )
        marked_loader, _, test_loader = svhn_dataloaders(
            batch_size=args.batch_size,
            data_dir=args.data,
            num_workers=args.workers,
            class_to_replace=args.class_to_replace,
            num_indexes_to_replace=args.num_indexes_to_replace,
            indexes_to_replace=args.indexes_to_replace,
            seed=args.seed,
            only_mark=True,
            shuffle=True,
        )
        if args.imagenet_arch:
            model = model_dict[args.arch](num_classes=classes, imagenet=True)
        else:
            model = model_dict[args.arch](num_classes=classes)

        model.normalize = normalization
        return model, train_full_loader, val_loader, test_loader, marked_loader
    elif args.dataset == "cifar100":
        classes = 100
        normalization = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4866, 0.4409], std=[0.2673, 0.2564, 0.2762]
        )
        train_full_loader, val_loader, _ = cifar100_dataloaders(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers
        )
        marked_loader, _, test_loader = cifar100_dataloaders(
            batch_size=args.batch_size,
            data_dir=args.data,
            num_workers=args.workers,
            class_to_replace=args.class_to_replace,
            num_indexes_to_replace=args.num_indexes_to_replace,
            indexes_to_replace=args.indexes_to_replace,
            seed=args.seed,
            only_mark=True,
            shuffle=True,
            no_aug=args.no_aug,
        )
        if args.imagenet_arch:
            model = model_dict[args.arch](num_classes=classes, imagenet=True)
        else:
            model = model_dict[args.arch](num_classes=classes)
        model.normalize = normalization
        return model, train_full_loader, val_loader, test_loader, marked_loader
    elif args.dataset == "TinyImagenet":
        classes = 200
        normalization = NormalizeByChannelMeanStd(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        train_full_loader, val_loader, test_loader = TinyImageNet(args).data_loaders(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers
        )
        # train_full_loader, val_loader, test_loader =None, None,None
        marked_loader, _, _ = TinyImageNet(args).data_loaders(
            batch_size=args.batch_size,
            data_dir=args.data,
            num_workers=args.workers,
            class_to_replace=args.class_to_replace,
            num_indexes_to_replace=args.num_indexes_to_replace,
            indexes_to_replace=args.indexes_to_replace,
            seed=args.seed,
            only_mark=True,
            shuffle=True,
        )
        if args.imagenet_arch:
            model = model_dict[args.arch](num_classes=classes, imagenet=True)
        else:
            model = model_dict[args.arch](num_classes=classes)

        model.normalize = normalization
        return model, train_full_loader, val_loader, test_loader, marked_loader

    elif args.dataset == "imagenet":
        classes = 1000
        normalization = NormalizeByChannelMeanStd(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        train_ys = torch.load(args.train_y_file)
        val_ys = torch.load(args.val_y_file)
        if args.arch == "convnext_tiny" and args.unlearn == "retrain":
            model = convnext_tiny(pretrained=True, in_22k=True)
        else:
            model = model_dict[args.arch](num_classes=classes, imagenet=True)
        
        model.normalize = normalization
            
        train_subset_indices = torch.ones_like(train_ys)
        
        if args.class_to_replace is None:
            class_to_replace = None
        
        elif type(args.class_to_replace) != list:
            if not args.class_to_replace.isdigit():
                class_file = f"./class_to_replace/{args.class_to_replace}.txt"
                with open(class_file, "r") as f:
                    class_to_replace = [int(line.strip()) for line in f if line.strip()]
        
        # when train the model
        if class_to_replace is None and args.num_indexes_to_replace is None:
            train_subset_indices = None
            
        elif args.num_indexes_to_replace:
            total_samples = len(train_ys)
            num_to_replace = min(args.num_indexes_to_replace, total_samples)
            replace_indices = torch.randperm(total_samples)[:num_to_replace]
            train_subset_indices[replace_indices] = 0
            
        elif class_to_replace:
            for class_id in class_to_replace:
                class_id = int(class_id)
                train_class_indices = (train_ys == class_id).nonzero().squeeze()
                train_subset_indices[train_class_indices] = 0
        
        # divide validation set into retain and forget
        val_subset_indices = torch.ones_like(val_ys)
        
        if class_to_replace is None and args.num_indexes_to_replace is None:
            val_subset_indices = None
            
        elif args.num_indexes_to_replace:
            total_samples = len(val_ys)
            num_to_replace = min(args.num_indexes_to_replace, total_samples)
            replace_indices = torch.randperm(total_samples)[:num_to_replace]
            val_subset_indices[replace_indices] = 0
            
        elif class_to_replace:
            for class_id in class_to_replace:
                class_id = int(class_id)
                val_class_indices = (val_ys == class_id).nonzero().squeeze()
                val_subset_indices[val_class_indices] = 0

        loaders = prepare_data(
            dataset="imagenet",
            batch_size=args.batch_size,
            train_subset_indices=train_subset_indices,
            val_subset_indices=val_subset_indices,
            args=args,
            data_path=args.data_dir
        )
        retain_loader = loaders["train"]
        val_retain_loader = loaders["val_train"]
        if train_subset_indices is None:
            forget_loader = None
            return model, retain_loader, val_retain_loader
        else:
            forget_loader = loaders["fog"]
            val_forget_loader = loaders["val_fog"]
            return model, retain_loader, forget_loader, val_retain_loader, val_forget_loader
        

    elif args.dataset == "cifar100_no_val":
        classes = 100
        normalization = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4866, 0.4409], std=[0.2673, 0.2564, 0.2762]
        )
        train_set_loader, val_loader, test_loader = cifar100_dataloaders_no_val(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers
        )

    elif args.dataset == "cifar10_no_val":
        classes = 10
        normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
        )
        train_set_loader, val_loader, test_loader = cifar10_dataloaders_no_val(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers
        )

    else:
        raise ValueError("Dataset not support!")

    if args.imagenet_arch:
        model = model_dict[args.arch](num_classes=classes, imagenet=True)
    else:
        model = model_dict[args.arch](num_classes=classes)

    model.normalize = normalization
    return model, train_set_loader, val_loader, test_loader


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class NormalizeByChannelMeanStd(torch.nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return self.normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return "mean={}, std={}".format(self.mean, self.std)

    def normalize_fn(self, tensor, mean, std):
        """Differentiable version of torchvision.functional.normalize"""
        # here we assume the color channel is in at dim=1
        mean = mean[None, :, None, None]
        std = std[None, :, None, None]
        return tensor.sub(mean).div(std)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def run_commands(gpus, commands, call=False, dir="commands", shuffle=True, delay=0.5):
    if len(commands) == 0:
        return
    if os.path.exists(dir):
        shutil.rmtree(dir)
    if shuffle:
        random.shuffle(commands)
        random.shuffle(gpus)
    os.makedirs(dir, exist_ok=True)

    fout = open("stop_{}.sh".format(dir), "w")
    print("kill $(ps aux|grep 'bash " + dir + "'|awk '{print $2}')", file=fout)
    fout.close()

    n_gpu = len(gpus)
    for i, gpu in enumerate(gpus):
        i_commands = commands[i::n_gpu]
        if len(i_commands) == 0:
            continue
        prefix = "CUDA_VISIBLE_DEVICES={} ".format(gpu)

        sh_path = os.path.join(dir, "run{}.sh".format(i))
        fout = open(sh_path, "w")
        for com in i_commands:
            print(prefix + com, file=fout)
        fout.close()
        if call:
            os.system("bash {}&".format(sh_path))
            time.sleep(delay)


def get_loader_from_dataset(dataset, batch_size, seed=1, shuffle=True):
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=0, pin_memory=True, shuffle=shuffle
    )


def get_unlearn_loader(marked_loader, args):
    forget_dataset = copy.deepcopy(marked_loader.dataset)
    marked = forget_dataset.targets < 0
    forget_dataset.data = forget_dataset.data[marked]
    forget_dataset.targets = -forget_dataset.targets[marked] - 1
    forget_loader = get_loader_from_dataset(
        forget_dataset, batch_size=args.batch_size, seed=args.seed, shuffle=True
    )
    retain_dataset = copy.deepcopy(marked_loader.dataset)
    marked = retain_dataset.targets >= 0
    retain_dataset.data = retain_dataset.data[marked]
    retain_dataset.targets = retain_dataset.targets[marked]
    retain_loader = get_loader_from_dataset(
        retain_dataset, batch_size=args.batch_size, seed=args.seed, shuffle=True
    )
    print("datasets length: ", len(forget_dataset), len(retain_dataset))
    return forget_loader, retain_loader


def get_poisoned_loader(poison_loader, unpoison_loader, test_loader, poison_func, args):
    poison_dataset = copy.deepcopy(poison_loader.dataset)
    poison_test_dataset = copy.deepcopy(test_loader.dataset)

    poison_dataset.data, poison_dataset.targets = poison_func(
        poison_dataset.data, poison_dataset.targets
    )
    poison_test_dataset.data, poison_test_dataset.targets = poison_func(
        poison_test_dataset.data, poison_test_dataset.targets
    )

    full_dataset = torch.utils.data.ConcatDataset(
        [unpoison_loader.dataset, poison_dataset]
    )

    poisoned_loader = get_loader_from_dataset(
        poison_dataset, batch_size=args.batch_size, seed=args.seed, shuffle=False
    )
    poisoned_full_loader = get_loader_from_dataset(
        full_dataset, batch_size=args.batch_size, seed=args.seed, shuffle=True
    )
    poisoned_test_loader = get_loader_from_dataset(
        poison_test_dataset, batch_size=args.batch_size, seed=args.seed, shuffle=False
    )

    return poisoned_loader, unpoison_loader, poisoned_full_loader, poisoned_test_loader
