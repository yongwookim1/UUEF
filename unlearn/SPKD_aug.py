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


def apply_crop_resize(img, min_ratio=0.9, max_ratio=0.9, size=224):
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
    
    # create new distill loader with augmented transforms
    distill_dataset = data_loaders["retain"].dataset
    retain_transform = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    distill_dataset.transform = retain_transform
    distill_loader = torch.utils.data.DataLoader(
        distill_dataset,
        batch_size=data_loaders["retain"].batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    features = []

    # switch mode
    model.train()

    start = time.time()
    if args.imagenet_arch:
        # unlearning phase
        print("Unlearning phase")
        for i, data in enumerate(tqdm(forget_loader)):
            image, target = get_x_y_from_data_dict(data, device)

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
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
        for i, data in enumerate(distill_loader):
            image, target = get_x_y_from_data_dict(data, device)
            
            aug_image = image
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
            
            # compute similarity matrices
            f_s, f_t = features_s[0], features_t[0]
            features.append(f_s.cpu())
            b = f_s.size(0)
            f_s_flat = f_s.view(b, -1)
            f_t_flat = f_t.view(b, -1)
            
            similarity_s = torch.mm(f_s_flat, f_s_flat.t())
            similarity_t = torch.mm(f_t_flat, f_t_flat.t())
            
            similarity_s = F.normalize(similarity_s, p=2, dim=1)
            similarity_t = F.normalize(similarity_t, p=2, dim=1)
            
            # compute similarity loss
            similarity_loss = torch.norm(similarity_s - similarity_t, p='fro') / (b * b)

            features_s.clear()
            features_t.clear()
            
            for hook in hooks:
                hook.remove()
            for hook in hooks_t:
                hook.remove()
            
            ce_loss = criterion(output, target)
            loss = 10 * ce_loss + 10 * similarity_loss
            
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

    return top1.avg, features