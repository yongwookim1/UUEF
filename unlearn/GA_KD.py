import sys
import time
import copy

import torch
import torch.nn.functional as F
import utils

from .impl import iterative_unlearn

sys.path.append(".")
from imagenet import get_x_y_from_data_dict


@iterative_unlearn
def GA_KD(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    # store initial model state at the beginning of unlearning (epoch 0)
    if not hasattr(GA_KD, 'original_model'):
        GA_KD.original_model = copy.deepcopy(model)
        GA_KD.original_model.eval()
        for param in GA_KD.original_model.parameters():
            param.requires_grad = False
    
    original_model = GA_KD.original_model
    
    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]
    distill_loader = retain_loader
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # switch to train mode
    model.train()

    start = time.time()
    if args.imagenet_arch:
        # unlearning phase
        print("Unlearning phase")
        for i, data in enumerate(forget_loader):
            image, target = get_x_y_from_data_dict(data, device)
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(forget_loader), args=args
                )

            # compute output
            output_clean = model(image)

            loss = -criterion(output_clean, target)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            output = output_clean.float()
            loss = loss.float()
            prec1 = utils.accuracy(output.data, target)[0]

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

        print("Restore phase")
        
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
        
        for i, data in enumerate(distill_loader):
            image, target = get_x_y_from_data_dict(data, device)

            # compute output
            output_u = model(image)
            with torch.no_grad():
                output_o = original_model(image)
            
            # combine KD loss and CE loss
            temperature = 2
            kd_loss = F.kl_div(
                F.log_softmax(output_u/temperature, dim=1),
                F.softmax(output_o/temperature, dim=1),
                reduction='batchmean'
            ) * (temperature * temperature)
            ce_loss = criterion(output_u, target)
            
            loss = 10 * kd_loss + 10 * ce_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            output = output_u.float()
            loss = loss.float()
            prec1 = utils.accuracy(output.data, target)[0]

            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Time {3:.2f}".format(
                        epoch, i, len(distill_loader), end - start, loss=losses, top1=top1
                    )
                )
                start = time.time()

    print("train_accuracy {top1.avg:.3f}".format(top1=top1))

    return top1.avg
