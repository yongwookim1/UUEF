import sys
import time
import copy

import torch
import torch.nn.functional as F
import utils
import random
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm

from .impl import iterative_unlearn
sys.path.append(".")
from imagenet import get_x_y_from_data_dict


def apply_crop_resize(img, min_ratio=0.5, max_ratio=0.5, size=224):
    """random crop and resize augmentation with configurable ratios"""
    min_size = int(min_ratio * size)
    max_size = int(max_ratio * size)
    crop_size = random.randint(min_size, max_size)
    crop = transforms.RandomCrop(crop_size)
    resize = transforms.Resize(size, antialias=True)
    result = resize(crop(img))
    return result


@iterative_unlearn
def SPKD_aug(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    if not hasattr(SPKD_aug, 'original_model'):
        SPKD_aug.original_model = copy.deepcopy(model)
        SPKD_aug.original_model.eval()
        for param in SPKD_aug.original_model.parameters():
            param.requires_grad = False
    
    original_model = SPKD_aug.original_model
    
    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]
    distill_loader = forget_loader
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # switch mode
    model.train()

    start = time.time()
    if args.imagenet_arch:
        # unlearning phase
        print("Unlearning phase")
        for i, data in enumerate(tqdm(forget_loader)):
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
        for i, data in enumerate(tqdm(distill_loader)):
            image, target = get_x_y_from_data_dict(data, device)

            # apply augmentation
            aug_image = apply_crop_resize(image)

            # extract feature map
            features_s = []
            features_t = []
            
            def hook_fn(module, input, output):
                features_s.append(output)
                
            def hook_fn_t(module, input, output):
                features_t.append(output)
            
            hooks = []
            hooks_t = []
            
            hooks.append(model.avgpool.register_forward_hook(hook_fn))
            hooks_t.append(original_model.avgpool.register_forward_hook(hook_fn_t))
            
            output = model(aug_image)
            with torch.no_grad():
                _ = original_model(aug_image)
            
            f_s, f_t = features_s[0], features_t[0]
            f_s_flat = f_s.view(f_s.size(0), -1)
            f_t_flat = f_t.view(f_t.size(0), -1)
            
            # compute similarity loss
            similarity_loss = F.mse_loss(f_s_flat, f_t_flat)
            
            features_s.clear()
            features_t.clear()
            
            for hook in hooks:
                hook.remove()
            for hook in hooks_t:
                hook.remove()

            ce_loss = criterion(output, target)
            loss = similarity_loss

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