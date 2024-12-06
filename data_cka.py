from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
import os
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import math

import arg_parser
import utils
from CKA.CKA import CudaCKA


def load_model(pretrained_model_path, device):
    print(f"\nLoading model: {pretrained_model_path}")
    model = models.resnet50(weights=None)
    checkpoint = torch.load(pretrained_model_path, map_location=device)
    if "state_dict" in checkpoint.keys():
        checkpoint = checkpoint["state_dict"]
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    state_dict = {
        k: v for k, v in state_dict.items() if not k.startswith("normalize.")
    }
    model.load_state_dict(state_dict, strict=True)
    print(f"Model loading complete: {pretrained_model_path}")
    return model


def hook_fn_o(module, input, output):
    features_o.append(output)


def hook_fn_r(module, input, output):
    features_r.append(output)


def rand_bbox(img_shape, lam):
    W = img_shape[2]
    H = img_shape[3]
    
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2


def add_gaussian_noise(img, sigma):
    noise = torch.randn_like(img) * sigma
    noisy_img = img + noise
    return torch.clamp(noisy_img, 0, 1)


def mix_forget_retain(forget_img, retain_img, ratio=0.5, mode='mixup'):
    """random crop and resize augmentation with configurable ratios"""
    batch_size = min(forget_img.size(0), retain_img.size(0))
    forget_img = forget_img[:batch_size]
    retain_img = retain_img[:batch_size]
    
    if mode.lower() == 'mixup':
        mixed_img = ratio * forget_img + (1 - ratio) * retain_img
    elif mode.lower() == 'cutmix':
        mixed_img = retain_img.clone()
        bbx1, bby1, bbx2, bby2 = rand_bbox(retain_img.shape, ratio)
        mixed_img[:, :, bbx1:bbx2, bby1:bby2] = forget_img[:, :, bbx1:bbx2, bby1:bby2]
    else:
        raise ValueError(f"Unknown mixing mode: {mode}")
    return mixed_img


def apply_crop_resize(img, min_ratio=0.3, max_ratio=0.6, size=224):
    """random crop and resize augmentation with configurable ratios"""
    min_size = int(min_ratio * size)
    max_size = int(max_ratio * size)
    crop_size = random.randint(min_size, max_size)
    crop = transforms.RandomCrop(crop_size)
    resize = transforms.Resize(size)
    return resize(crop(img))


def apply_color_distortion(img, p_drop=0.3):
    """color channel dropping"""
    if random.random() < p_drop:
        channel = random.randint(0, 2)
        img[:, channel, :, :] = 0
    return img


def apply_color_jitter(img):
    """color jittering"""
    jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
    return jitter(img)


def apply_rotation(img):
    """rotate by 90, 180, or 270 degrees"""
    angle = random.choice([90, 180, 270])
    return TF.rotate(img, angle)


def apply_cutout(img, ratio=0.5):
    """apply cutout augmentation"""
    h, w = img.shape[2], img.shape[3]
    length_h = int(h * ratio)
    length_w = int(w * ratio)
    y = random.randint(0, h - length_h)
    x = random.randint(0, w - length_w) 
    img_copy = img.clone()
    img_copy[:, :, y:y+length_h, x:x+length_w] = 0
    return img_copy


def apply_gaussian_blur(img, kernel_size=5, sigma=2):
    """apply gaussian blur"""
    blur = transforms.GaussianBlur(kernel_size, sigma)
    return blur(img)


def main():
    args = arg_parser.parse_args()
    data = args.data_type
    seed = 2
    utils.setup_seed(seed)
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    model, retain_loader, forget_loader, val_loader = utils.setup_model_dataset(args)
    
    unlearn_data_loaders = OrderedDict(
        retain=retain_loader, forget=forget_loader, val=val_loader
    )
    
    pretrained_model_paths = [
        "./pretrained_model/0model_SA_best159.pth.tar",
        "./pretrained_model/retraincheckpoint100.pth.tar",
    ]
    
    # validate data choice
    if data not in unlearn_data_loaders:
        raise ValueError(f"Invalid data type: {data}. Must be one of {list(unlearn_data_loaders.keys())}")
    
    data_loader = unlearn_data_loaders[data]
    
    original_model = load_model(pretrained_model_paths[0], device).to(device)
    retrained_model = load_model(pretrained_model_paths[1], device).to(device)
    
    original_model.eval()
    retrained_model.eval()
    
    global features_o, features_r
    
    # define augmentation methods
    augmentation_methods = {
        'original': lambda x: x,
        'gaussian': lambda x: add_gaussian_noise(x, sigma=0.1),
        'mixup': lambda x, y: mix_forget_retain(x, y, ratio=0.5, mode='mixup'),
        'cutmix': lambda x, y: mix_forget_retain(x, y, ratio=0.5, mode='cutmix'),
        'crop_resize': apply_crop_resize,
        'color_distortion': apply_color_distortion,
        'color_jitter': apply_color_jitter,
        'rotation': apply_rotation,
        'cutout': apply_cutout,
        'gaussian_blur': apply_gaussian_blur
    }

    # choose method to use
    method = 'original'  # change this to use different augmentations
    
    linear_cka = kernel_cka = linear_check = kernel_check = 0
    
    for i, data in enumerate(tqdm(data_loader)):
        mix_loader = retain_loader if args.data_type == "forget" else forget_loader
        img, _ = data
        img = img.to(device)
        
        # apply augmentation
        if method in ['mixup', 'cutmix']:
            # for methods that require two images
            try:
                mix_data = next(iter(mix_loader))
                mix_image, _ = mix_data
                mix_image = mix_image.to(device)
                aug_img = augmentation_methods[method](img, mix_image)
            except StopIteration:
                continue
        else:
            # for single image augmentations
            aug_img = augmentation_methods[method](img)
        
        features_o = []
        features_r = []
        
        hooks_o = []
        hooks_r = []
        
        hooks_o.append(original_model.avgpool.register_forward_hook(hook_fn_o))
        hooks_r.append(retrained_model.avgpool.register_forward_hook(hook_fn_r))
        
        with torch.no_grad():
            original_output = original_model(aug_img)
            retrained_output = retrained_model(aug_img)
            
            f_o, f_r = features_o[0], features_r[0]
            f_o = f_o.view(f_o.size(0), -1)
            f_r = f_r.view(f_r.size(0), -1)
            
            cuda_cka = CudaCKA(device)
            linear_cka += cuda_cka.linear_CKA(f_o, f_r)
            linear_check += cuda_cka.linear_CKA(f_o, f_o)
            kernel_cka += cuda_cka.kernel_CKA(f_o, f_r)
            kernel_check += cuda_cka.kernel_CKA(f_r, f_r)
    
        for hook in hooks_o:
            hook.remove()
        for hook in hooks_r:
            hook.remove()
    
    print(f"\n=== Results for {method} ===")
    print(f"Linear CKA: {linear_cka / len(data_loader):.4f}")
    print(f"Kernel CKA: {kernel_cka / len(data_loader):.4f}")
    print(f"Linear CKA check: {linear_check / len(data_loader):.4f}")
    print(f"Kernel CKA check: {kernel_check / len(data_loader):.4f}")


if __name__ == "__main__":
    main()
