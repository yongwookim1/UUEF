import sys
import time
import copy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import utils

from .impl import iterative_unlearn

sys.path.append(".")
from imagenet import get_x_y_from_data_dict


@iterative_unlearn
def SCRUB(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    # store initial model state at the beginning of unlearning (epoch 0)
    if not hasattr(SCRUB, 'original_model'):
        SCRUB.original_model = copy.deepcopy(model)
        SCRUB.original_model.eval()
        for param in SCRUB.original_model.parameters():
            param.requires_grad = False
    
    original_model = SCRUB.original_model
    
    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # switch mode
    model.train()

    start = time.time()
    
    # hyperparameters
    max_steps = 100
    min_steps = 100
    forget_batch_size = 256
    retain_batch_size = 256
    
    forget_loader = DataLoader(forget_loader.dataset,
                               batch_size=forget_batch_size,
                               shuffle=True,
                               num_workers=2)
    
    retain_loader = DataLoader(retain_loader.dataset,
                               batch_size=retain_batch_size,
                               shuffle=True,
                               num_workers=4)
    
    # unlearning phase
    print("Unlearning phase")
    for i, data in enumerate(forget_loader):
        if i == max_steps:
            break
        image, target = get_x_y_from_data_dict(data, device)

        with torch.no_grad():
            output_original = original_model(image)
        output_unlearned = model(image)
            
        # KL-divergence
        loss = -1 * F.kl_div(F.log_softmax(output_unlearned, dim=1), F.softmax(output_original, dim=1), reduction='batchmean')
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        output = output_unlearned.float()
        loss = loss.float()
        prec1 = utils.accuracy(output_unlearned.data, target)[0]

        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

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
    
    # restore phase
    print("Restore phase")
    for i, data in enumerate(retain_loader):
        if i == min_steps:
            break
        image, target = get_x_y_from_data_dict(data, device)
        
        with torch.no_grad():
            output_original = original_model(image)
        output_unlearned = model(image)

        # KL-divergence
        kld_loss = F.kl_div(F.log_softmax(output_unlearned, dim=1), F.softmax(output_original, dim=1), reduction='batchmean')
        ce_loss = criterion(output_unlearned, target)
        
        loss = kld_loss + ce_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output_unlearned.float()
        loss = loss.float()
        prec1 = utils.accuracy(output_unlearned.data, target)[0]

        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if (i + 1) % args.print_freq == 0:
            end = time.time()
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                "Time {3:.2f}".format(
                    epoch, i, len(retain_loader), end - start, loss=losses, top1=top1
                )
            )
            start = time.time()

    print("train_accuracy {top1.avg:.3f}".format(top1=top1))

    return top1.avg
