import sys
import time
import os
import copy

import torch
import utils
from tqdm.auto import tqdm

from .impl import iterative_unlearn

sys.path.append(".")
from imagenet import get_x_y_from_data_dict


def gaussian_kernel(x, y, sigma=1.0):
    """calculate gaussian kernel between two sets of features"""
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    
    x = x.unsqueeze(1).expand(x_size, y_size, dim)
    y = y.unsqueeze(0).expand(x_size, y_size, dim)
    
    kernel_val = torch.exp(-torch.sum((x - y)**2, dim=2) / (2 * sigma**2))
    return kernel_val


def calculate_mmd(x, y, sigma=1.0):
    """calculate maximum mean discrepancy between two sets of features"""
    x_kernel = gaussian_kernel(x, x, sigma)
    y_kernel = gaussian_kernel(y, y, sigma)
    xy_kernel = gaussian_kernel(x, y, sigma)
    
    mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
    return mmd


def calculate_retain_features(model, retain_loader, device):
    """calculate and store features of retain set"""
    model.eval()
    model = model.to(device)
    
    features = []
    hooks = []
    
    def hook_fn(module, input, output):
        features.append(output.squeeze())
        
    hooks.append(model.avgpool.register_forward_hook(hook_fn))
    
    with torch.no_grad():
        for images, _ in tqdm(retain_loader):
            images = images.to(device)
            model(images)
    
    features = torch.cat(features, dim=0)
    
    for hook in hooks:
        hook.remove()
    
    return features.to(device)


def perturb_images(model, images, retain_features):
    """perturb images to maximize MMD distance"""
    perturbed_images = images.clone().detach().requires_grad_(True)
    
    for _ in range(10):
        # calculate features for current perturbed images
        features = []
        def hook_fn(module, input, output):
            features.append(output.squeeze())
        hook = model.avgpool.register_forward_hook(hook_fn)
        model(perturbed_images)
        hook.remove()
        curr_features = features[0]
        
        # calculate MMD distance
        mmd_dist = calculate_mmd(curr_features, retain_features)
        loss = -0.1 * mmd_dist  # negative because we want to maximize distance
        loss.backward()
        
        # update images using gradient
        grad = perturbed_images.grad
        with torch.no_grad():
            perturbed_images = perturbed_images + 1e-5 * grad.sign()
            perturbed_images = torch.clamp(perturbed_images, 0, 1)
        
        perturbed_images.grad = None
        perturbed_images = perturbed_images.detach().requires_grad_(True)
    return perturbed_images


@iterative_unlearn
def GA_mmd(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    # store initial model state at the beginning of unlearning (epoch 0)
    if not hasattr(GA_mmd, 'original_model'):
        GA_mmd.original_model = copy.deepcopy(model)
        GA_mmd.original_model.eval()
        for param in GA_mmd.original_model.parameters():
            param.requires_grad = False
    
    original_model = GA_mmd.original_model
    
    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]
    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    
    stats_path = "./stats/retain_features.pth"
    if os.path.exists(stats_path):
        if not hasattr(GA_mmd, 'retain_features'):
            GA_mmd.retain_features = torch.load(stats_path, map_location=device)
        retain_features = GA_mmd.retain_features
    else:
        if not hasattr(GA_mmd, 'retain_features'):
            GA_mmd.retain_features = calculate_retain_features(model, retain_loader, device)
        retain_features = GA_mmd.retain_features
        
        os.makedirs("./stats", exist_ok=True)
        torch.save(retain_features.cpu(), stats_path)
    
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    model.train()
    
    start = time.time()
    for i, (images, targets) in enumerate(tqdm(forget_loader)):
        images = images.to(device)
        targets = targets.to(device)
        
        # perturb images to maximize MMD distance
        perturbed_images = perturb_images(original_model, images, retain_features)
        
        # regular training step with perturbed images
        outputs = model(perturbed_images)
        loss = -criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure accuracy and record loss
        prec1 = utils.accuracy(outputs.data, targets)[0]
        losses.update(loss.item(), images.size(0))
        top1.update(prec1.item(), images.size(0))
        
        if (i + 1) % args.print_freq == 0:
            end = time.time()
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                "Time {3:.2f}".format(
                    epoch, i, len(forget_loader), end - start, loss=losses, top1=top1
                )
            )
            start = time.time()
    
    print("train_accuracy {top1.avg:.3f}".format(top1=top1))
    return top1.avg
