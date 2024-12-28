import sys
import time

import torch
from torch.utils.data import DataLoader
import utils

from .impl import iterative_unlearn

sys.path.append(".")
from imagenet import get_x_y_from_data_dict


@iterative_unlearn
def CU(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]
    
    retain_loader = DataLoader(retain_loader.dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=4)
    
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    model.train()
    start = time.time()

    if args.imagenet_arch:
        device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
        
        for i, (forget_data, retain_data) in enumerate(zip(forget_loader, retain_loader)):
            forget_images, forget_targets = get_x_y_from_data_dict(forget_data, device)
            retain_images, retain_targets = get_x_y_from_data_dict(retain_data, device)
            
            # add hook to capture avgpool features
            features = []
            
            def hook_fn(module, input, output):
                features.append(output)
                
            hooks = []
            
            # register the hook on the avgpool layer
            hooks.append(model.avgpool.register_forward_hook(hook_fn))
            
            # forward pass will trigger the hook
            forget_outputs = model(forget_images)
            retain_outputs = model(retain_images)
            
            ce_loss = criterion(retain_outputs, retain_targets)
            
            # get embeddings and normalize
            forget_embeddings = features[0].view(forget_images.size(0), -1)
            forget_normalized = torch.nn.functional.normalize(forget_embeddings, dim=1)
            
            features.clear()
            
            for hook in hooks:
                hook.remove()
            
            # hyperparameters for contrastive loss
            temperature = 1.0  # controls the scaling of similarity scores
            margin = 0.0      # minimum distance between negative pairs
            pos_weight = 1.0  # weight for positive pair loss
            neg_weight = 1.0  # weight for negative pair loss

            # calculate cosine similarity between all pairs of forget samples
            similarity_matrix = torch.matmul(forget_normalized, forget_normalized.T) / temperature
            
            # create masks for positive and negative pairs
            labels_matrix = forget_targets.unsqueeze(0) == forget_targets.unsqueeze(1)  # identify same-class pairs
            mask = ~torch.eye(forget_targets.size(0), device=device).bool()  # exclude self-pairs
            pos_mask = labels_matrix & mask  # pairs with same class (positive pairs)
            neg_mask = ~labels_matrix & mask  # pairs with different classes (negative pairs)
            
            # calculate log probabilities for NCE loss
            exp_sim = torch.exp(similarity_matrix)
            log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
            
            contrast_loss = (-log_prob * neg_mask.float()).sum(dim=1) / neg_mask.sum(dim=1).clamp(min=1)
            contrast_loss = contrast_loss.mean()

            total_loss = ce_loss + contrast_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            prec1 = utils.accuracy(forget_outputs.data, forget_targets)[0]
            losses.update(total_loss.item(), forget_images.size(0))
            top1.update(prec1.item(), forget_images.size(0))

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print(
                    "epoch: [{0}][{1}/{2}]\t"
                    "loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                    "time {3:.2f}".format(
                        epoch, i, len(forget_loader), end - start, loss=losses, top1=top1
                    )
                )
                start = time.time()
    return top1.avg
