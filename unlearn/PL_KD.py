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
def PL_KD(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    # store initial model state at the beginning of unlearning (epoch 0)
    if not hasattr(PL_KD, 'original_model'):
        PL_KD.original_model = copy.deepcopy(model)
        PL_KD.original_model.eval()
        for param in PL_KD.original_model.parameters():
            param.requires_grad = False
    
    original_model = PL_KD.original_model
    
    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # switch mode
    model.train()
    
    if type(args.class_to_replace) != list:
        if not args.class_to_replace.isdigit():
            class_file = f"./class_to_replace/{args.class_to_replace}.txt"
            with open(class_file, "r") as f:
                class_to_replace = [int(line.strip()) for line in f if line.strip()]

    start = time.time()
    if args.imagenet_arch:
        # Unlearning phase
        print("Unlearning phase")
        for i, data in enumerate(forget_loader):
            image, target = get_x_y_from_data_dict(data, device)
            
            with torch.no_grad():
                output_o = original_model(image)
            output_o_modified = output_o.clone()
            output_o_modified[:, class_to_replace] = float('-inf')
            remain_target = output_o_modified.argmax(dim=1)
            
            output = model(image)
            loss = criterion(output, remain_target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            output = output.float()
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

        # Restore phase with KD
        print("Restore phase")
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
        
        for i, data in enumerate(retain_loader):
            image, target = get_x_y_from_data_dict(data, device)

            output_s = model(image)
            with torch.no_grad():
                output_t = original_model(image)
            
            temperature = 2
            kd_loss = F.kl_div(
                F.log_softmax(output_s/temperature, dim=1),
                F.softmax(output_t/temperature, dim=1),
                reduction='batchmean'
            ) * (temperature * temperature)
            
            ce_loss = criterion(output_s, target)
            loss = 1 * kd_loss + 1 * ce_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            output = output_s.float()
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
                        epoch, i, len(retain_loader), end - start, loss=losses, top1=top1
                    )
                )
                start = time.time()

    print("train_accuracy {top1.avg:.3f}".format(top1=top1))
    return top1.avg 