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


def calculate_retain_statistics(model, retain_loader, device):
    """calculate mean and covariance matrix of retain set embeddings"""
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
    mean = features.mean(dim=0)
    centered = features - mean

    cov = torch.mm(centered.t(), centered) / (features.size(0) - 1)
    identity = torch.eye(cov.shape[0], device=device)
    inv_cov = torch.linalg.inv(cov + identity * 1e-6)
    
    for hook in hooks:
        hook.remove()
    
    return mean.to(device), inv_cov.to(device)


def perturb_images(model, images, retain_mean, retain_inv_cov):
        """perturb images to maximize Mahalanobis distance"""
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
            
            # calculate Mahalanobis distance
            centered = curr_features - retain_mean
            mahalanobis_dist = torch.sqrt(torch.sum(torch.mm(centered, retain_inv_cov) * centered, dim=1))
            loss = -1 * mahalanobis_dist.mean()
            loss.backward()
            
            # update images using gradient
            grad = perturbed_images.grad
            with torch.no_grad():
                perturbed_images = perturbed_images + 1e-4 * grad.sign()
                perturbed_images = torch.clamp(perturbed_images, -1, 1)
            
            perturbed_images.grad = None
            perturbed_images = perturbed_images.detach().requires_grad_(True)
        return perturbed_images


@iterative_unlearn
def GA_pp(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    # store initial model state at the beginning of unlearning (epoch 0)
    if not hasattr(GA_pp, 'original_model'):
        GA_pp.original_model = copy.deepcopy(model)
        GA_pp.original_model.eval()
        for param in GA_pp.original_model.parameters():
            param.requires_grad = False
    
    original_model = GA_pp.original_model
    
    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]
    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    
    stats_path = "./stats/retain_stats.pth"
    if os.path.exists(stats_path):
        if not hasattr(GA_pp, 'retain_stats'):
            stats_dict = torch.load(stats_path, map_location=device)
            GA_pp.retain_stats = (stats_dict["retain_mean"], stats_dict["retain_inv_cov"])
        retain_mean, retain_inv_cov = GA_pp.retain_stats
        retain_mean = retain_mean.to(device)
        retain_inv_cov = retain_inv_cov.to(device)
    else:
        if not hasattr(GA_pp, 'retain_stats'):
            GA_pp.retain_stats = calculate_retain_statistics(model, retain_loader, device)
        retain_mean, retain_inv_cov = GA_pp.retain_stats
        retain_mean = retain_mean.to(device)
        retain_inv_cov = retain_inv_cov.to(device)
        
        os.makedirs("./stats", exist_ok=True)
        torch.save({"retain_mean": retain_mean.cpu(), "retain_inv_cov": retain_inv_cov.cpu()}, stats_path)
    
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    model.train()
    
    start = time.time()
    for i, (images, targets) in enumerate(tqdm(forget_loader)):
        images = images.to(device)
        targets = targets.to(device)
        
        # perturb images to maximize Mahalanobis distance
        perturbed_images = perturb_images(original_model, images, retain_mean, retain_inv_cov)
        
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
