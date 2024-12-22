import sys
import time
import copy
import random

import torch
import torch.nn.functional as F
import utils
import torchvision.transforms as transforms
from tqdm import tqdm

from .impl import iterative_unlearn
import os
sys.path.append(".")
from imagenet import get_x_y_from_data_dict
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)
from baseline.CKA.CKA import CudaCKA

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
def GA_CKA_SPKD(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    # store initial model state at the beginning of unlearning
    if not hasattr(GA_CKA_SPKD, 'original_model'):
        GA_CKA_SPKD.original_model = copy.deepcopy(model)
        GA_CKA_SPKD.original_model.eval()
    for param in GA_CKA_SPKD.original_model.parameters():
        param.requires_grad = False
        
    original_model = GA_CKA_SPKD.original_model
    
    forget_loader = data_loaders["forget"]
    
    # create new retain loader with augmented transforms
    retain_dataset = data_loaders["retain"].dataset
    retain_transform = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    retain_dataset.transform = retain_transform
    retain_loader = torch.utils.data.DataLoader(
        retain_dataset,
        batch_size=data_loaders["retain"].batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model.train()
    start = time.time()

    if args.imagenet_arch:
        # Unlearning phase
        print("Unlearning phase")
        for i, data in enumerate(tqdm(forget_loader)):
            image, target = get_x_y_from_data_dict(data, device)
            if epoch < args.warmup:
                utils.warmup_lr(epoch, i + 1, optimizer, one_epoch_step=len(forget_loader), args=args)

            output_clean = model(image)
            GA_loss = -criterion(output_clean, target)
            
            optimizer.zero_grad()
            GA_loss.backward()
            optimizer.step()

            output = output_clean.float()
            loss = GA_loss.float()
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

        # Restore phase with CKA and SPKD
        print("Restore phase")
        for i, data in enumerate(retain_loader):
            image, target = get_x_y_from_data_dict(data, device)
            
            # Apply augmentation
            aug_image = apply_crop_resize(image)

            features_u = []
            features_o = []
            def hook_fn_u(module, input, output):
                features_u.append(output)
            def hook_fn_o(module, input, output):
                features_o.append(output)

            hook_u = model.avgpool.register_forward_hook(hook_fn_u)
            hook_o = original_model.avgpool.register_forward_hook(hook_fn_o)
            
            output = model(aug_image)
            with torch.no_grad():
                _ = original_model(aug_image)
                
            hook_u.remove()
            hook_o.remove()

            # Calculate CKA similarity
            f_u = features_u[0].view(features_u[0].size(0), -1)
            f_o = features_o[0].view(features_o[0].size(0), -1)
            cka = CudaCKA(device)
            cka_similarity = cka.linear_CKA(f_u, f_o)
            
            # Calculate SPKD similarity matrices
            b = f_u.size(0)
            similarity_s = torch.mm(f_u, f_u.t())
            similarity_t = torch.mm(f_o, f_o.t())
            
            similarity_s = F.normalize(similarity_s, p=2, dim=1)
            similarity_t = F.normalize(similarity_t, p=2, dim=1)
            
            spkd_loss = torch.norm(similarity_s - similarity_t, p='fro') / (b * b)

            features_u.clear()
            features_o.clear()
            
            ce_loss = criterion(output, target)
            # Combine all losses
            loss = 10*ce_loss + 50*(1 - cka_similarity) + 10*spkd_loss

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
                        epoch, i, len(retain_loader), end - start, loss=losses, top1=top1
                    )
                )
                start = time.time()

    print("train_accuracy {top1.avg:.3f}".format(top1=top1))
    return top1.avg 