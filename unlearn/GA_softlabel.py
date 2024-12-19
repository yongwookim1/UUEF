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
def GA_softlabel(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    if not hasattr(GA_softlabel, 'original_model'):
        GA_softlabel.original_model = copy.deepcopy(model)
        GA_softlabel.original_model.eval()
        for param in GA_softlabel.original_model.parameters():
            param.requires_grad = False

    original_model = GA_softlabel.original_model
    
    train_loader = data_loaders["forget"]
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    if args.imagenet_arch:
        device = (
            torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
        )
        for i, data in enumerate(train_loader):
            image, target = get_x_y_from_data_dict(data, device)
            
            output_original = original_model(image)
            target_soft = F.softmax(output_original, dim=1)
            
            # compute output
            output_clean = model(image)
            
            loss = -criterion(output_clean, target_soft)
            
            # log_output = F.log_softmax(output_clean, dim=1)
            
            # # KL divergence loss
            # loss = -F.kl_div(log_output, target_soft, reduction='batchmean')
            
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
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
                        epoch, i, len(train_loader), end - start, loss=losses, top1=top1
                    )
                )
                start = time.time()

    print("train_accuracy {top1.avg:.3f}".format(top1=top1))

    return top1.avg
