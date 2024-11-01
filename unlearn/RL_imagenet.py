import time
from copy import deepcopy

import numpy as np
import torch
import utils

from .impl import iterative_unlearn
from imagenet import get_x_y_from_data_dict


@iterative_unlearn
def RL_imagenet(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    forget_loader = data_loaders["forget"]
    forget_dataset = deepcopy(forget_loader.dataset)
    
    original_targets = deepcopy(forget_dataset.dataset.targets)
    if mask is not None :
        print("mask on")
    
    if args.dataset == "imagenet":
        try:
            forget_dataset.targets = np.random.randint(0, args.num_classes, forget_dataset.targets.shape)
        except:
            forget_dataset.dataset.targets = np.random.randint(0, args.num_classes, len(forget_dataset.dataset.targets))
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
      
        # switch to train mode
        model.train()
      
        start = time.time()
        loader_len = len(forget_loader)
      
        if epoch < args.warmup:
            utils.warmup_lr(epoch, i+1, optimizer,
                            one_epoch_step=loader_len, args=args)
        
        for i, data in enumerate(forget_loader):
            image, target = get_x_y_from_data_dict(data, f"cuda:{int(args.gpu)}")
            target = torch.randint(0, args.num_classes, target.shape).cuda()
            
            # compute output
            output_clean = model(image)
            loss = criterion(output_clean, target)
            
            optimizer.zero_grad()
            loss.backward()
            
            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]
            
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
                        epoch, i, len(forget_loader), end - start, loss=losses, top1=top1
                    )
                )
                start = time.time()
                
    print("train_accuracy {top1.avg:.3f}".format(top1=top1))

    return top1.avg