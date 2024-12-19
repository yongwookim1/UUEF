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
def AKD_AL(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    # store initial model state at the beginning of unlearning (epoch 0)
    if not hasattr(AKD_AL, 'original_model'):
        AKD_AL.original_model = copy.deepcopy(model)
        AKD_AL.original_model.eval()
        for param in AKD_AL.original_model.parameters():
            param.requires_grad = False
    
    original_model = AKD_AL.original_model
    
    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]
    distill_loader = retain_loader
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # switch mode
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

        # restore phase
        print("Restore phase")
        for i, data in enumerate(distill_loader):
            image, target = get_x_y_from_data_dict(data, device)

            # extract feature maps
            features_s = []
            features_t = []
            
            def hook_fn(module, input, output):
                features_s.append(output)
                
            def hook_fn_t(module, input, output):
                features_t.append(output)
            
            hooks = []
            hooks_t = []
            
            # register hooks for all major feature extraction layers
            feature_layers = ['layer1', 'layer2', 'layer3', 'layer4', 'avgpool']
            for layer_name in feature_layers:
                hooks.append(getattr(model, layer_name).register_forward_hook(hook_fn))
                hooks_t.append(getattr(original_model, layer_name).register_forward_hook(hook_fn_t))
                    
            output = model(image)
            with torch.no_grad():
                _ = original_model(image)
            
            # compute similarity loss for all feature maps
            similarity_loss = 0
            for f_s, f_t in zip(features_s, features_t):
                f_s_flat = f_s.view(f_s.size(0), -1)
                f_t_flat = f_t.view(f_t.size(0), -1)
                similarity_loss += F.mse_loss(f_s_flat, f_t_flat)
            
            # normalize by number of feature maps
            similarity_loss /= len(features_s)
            
            features_s.clear()
            features_t.clear()
            
            for hook in hooks:
                hook.remove()
            for hook in hooks_t:
                hook.remove()

            ce_loss = criterion(output, target)
            loss = 10 * similarity_loss + 10 * ce_loss

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
                        epoch, i, len(distill_loader), end - start, loss=losses, top1=top1
                    )
                )
                start = time.time()

    print("train_accuracy {top1.avg:.3f}".format(top1=top1))

    return top1.avg 