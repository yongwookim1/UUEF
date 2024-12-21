"""
    setup model and datasets
"""


import copy
import os
import random

# from advertorch.utils import NormalizeByChannelMeanStd
import shutil
import sys
import time

import numpy as np
import torch
from torchvision import models
from torchvision import transforms
from dataset import *
from dataset import TinyImageNet
from imagenet import prepare_data
from models import *
import wandb
from CKA.CKA import CudaCKA
import main_knn

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


def evaluate_knn(model, args):
    setup_seed(2)
    
    train_loader, test_loader = main_knn.create_data_loaders(args)
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    train_features, train_labels = main_knn.extract_features(model, train_loader, device)
    test_features, test_labels = main_knn.extract_features(model, test_loader, device)
    
    knn_accuracy = main_knn.evaluate_knn(
        train_features,
        train_labels,
        test_features,
        test_labels,
        5,
    )
    print(f"kNN(k=5) accuracy: {knn_accuracy * 100:.2f}%")
    return knn_accuracy


class FeatureExtractor:
    def __init__(self):
        self.features = []
        
    def hook_fn(self, module, input, output):
        self.features.append(output)
        
    def clear(self):
        self.features = []


def evaluate_cka(model, retrained_model, data_loader, device, mode='all'):
    """
    compute CKA similarity between two models
    
    args:
        model: first model to compare
        retrained_model: second model to compare
        data_loader: dataLoader containing the dataset
        device: torch device to use
        mode: 'all' for all layers or 'avgpool' for only avgpool layer
    """
    model.eval()
    retrained_model.eval()

    # feature extractors
    original_extractor = FeatureExtractor()
    retrained_extractor = FeatureExtractor()
    cuda_cka = CudaCKA(device)

    if mode == 'all':
        layers = ['layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc']
        cka_result = {f"linear_cka{i}": 0 for i in range(len(layers))}
        cka_result.update({f"linear_check{i}": 0 for i in range(len(layers))})
    else:
        linear_cka = kernel_cka = linear_check = kernel_check = 0
        cuda_cka = CudaCKA(device)

    for data in tqdm(data_loader):
        img, _ = data
        img = img.to(device)

        if mode == 'all':
            hooks = [
                getattr(model, layer).register_forward_hook(original_extractor.hook_fn)
                for layer in layers
            ] + [
                getattr(retrained_model, layer).register_forward_hook(retrained_extractor.hook_fn)
                for layer in layers
            ]
        else:
            hooks = [
                model.avgpool.register_forward_hook(original_extractor.hook_fn),
                retrained_model.avgpool.register_forward_hook(retrained_extractor.hook_fn)
            ]

        with torch.no_grad():
            model(img)
            retrained_model(img)

            if mode == 'all':
                for i in range(len(layers)):
                    f_o = original_extractor.features[i].view(original_extractor.features[i].size(0), -1)
                    f_r = retrained_extractor.features[i].view(retrained_extractor.features[i].size(0), -1)
                    cka_result[f"linear_cka{i}"] += cuda_cka.linear_CKA(f_o, f_r)
                    cka_result[f"linear_check{i}"] += cuda_cka.linear_CKA(f_o, f_o)
            else:
                f_o = original_extractor.features[0].view(original_extractor.features[0].size(0), -1)
                f_r = retrained_extractor.features[0].view(retrained_extractor.features[0].size(0), -1)
                linear_cka += cuda_cka.linear_CKA(f_o, f_r)
                kernel_cka += cuda_cka.kernel_CKA(f_o, f_r)
                linear_check += cuda_cka.linear_CKA(f_o, f_o)
                kernel_check += cuda_cka.kernel_CKA(f_r, f_r)

        for hook in hooks:
            hook.remove()
        original_extractor.clear()
        retrained_extractor.clear()

    n = len(data_loader)
    if mode == 'all':
        final_results = {}
        cka_values = []
        for i, layer in enumerate(layers):
            avg_cka = cka_result[f"linear_cka{i}"] / n
            avg_check = cka_result[f"linear_check{i}"] / n
            cka_values.append(avg_cka)
            final_results[layer] = {
                'cka': avg_cka.cpu().numpy(),
            }
        
        return final_results
    else:
        return {
            'cka': linear_cka / n,
        }


def load_model(pretrained_model_path, device):
    print(f"\nLoading model: {pretrained_model_path}")
    model = models.resnet50(weights=None)
    checkpoint = torch.load(pretrained_model_path, map_location=device)
    if "state_dict" in checkpoint.keys():
        checkpoint = checkpoint["state_dict"]
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    state_dict = {k: v for k, v in state_dict.items() if not (k.startswith('normalize.'))}
    model.load_state_dict(state_dict, strict=True)
    print(f"Model loading complete: {pretrained_model_path}")
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
        model = model_dict[args.arch](num_classes=classes, imagenet=True)
        
        model.normalize = normalization
            
        train_subset_indices = torch.ones_like(train_ys)
        
        # when train the model
        if args.class_to_replace is None and args.num_indexes_to_replace is None:
            train_subset_indices = None
            
        elif args.num_indexes_to_replace:
            total_samples = len(train_ys)
            num_to_replace = min(args.num_indexes_to_replace, total_samples)
            replace_indices = torch.randperm(total_samples)[:num_to_replace]
            train_subset_indices[replace_indices] = 0
            
        elif args.class_to_replace:
            for class_id in args.class_to_replace:
                class_id = int(class_id)
                train_class_indices = (train_ys == class_id).nonzero().squeeze()
                train_subset_indices[train_class_indices] = 0

        loaders = prepare_data(
            dataset="imagenet",
            batch_size=args.batch_size,
            train_subset_indices=train_subset_indices,
            val_subset_indices=train_subset_indices,
            args=args,
            data_path=args.data_dir
        )
        retain_loader = loaders["train"]
        val_retain_loader = loaders["val_retain"]
        val_forget_loader = loaders["val_forget"]
        if train_subset_indices is None:
            forget_loader = None
            return model, retain_loader, val_retain_loader, val_forget_loader
        else:
            forget_loader = loaders["fog"]
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
        raise ValueError("Dataset not supprot yet !")
    # import pdb;pdb.set_trace()

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
