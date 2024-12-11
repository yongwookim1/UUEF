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
import matplotlib.pyplot as plt

import arg_parser
import utils
from CKA.CKA import CudaCKA


def hook_fn_o(module, input, output):
    features_o.append(output)


def hook_fn_r(module, input, output):
    features_r.append(output)


def rand_bbox(img_shape, lam):
    W = img_shape[2]
    H = img_shape[3]
    
    # ensure lam is between 0 and 1
    lam = np.clip(lam, 0.0, 1.0)
    
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # ensure minimum size of 1
    cut_w = max(1, min(cut_w, W-1))
    cut_h = max(1, min(cut_h, H-1))
    
    # adjust bounds to ensure valid random selection
    x_min = max(0, cut_w // 2)
    x_max = max(x_min + 1, W - cut_w // 2)
    y_min = max(0, cut_h // 2)
    y_max = max(y_min + 1, H - cut_h // 2)
    
    # generate random center position
    cx = np.random.randint(x_min, x_max)
    cy = np.random.randint(y_min, y_max)
    
    # calculate box coordinates with clipping
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
    """Mix forget and retain images using mixup or cutmix"""
    batch_size = min(forget_img.size(0), retain_img.size(0))
    forget_img = forget_img[:batch_size]
    retain_img = retain_img[:batch_size]
    
    if mode.lower() == 'mixup':
        mixed_img = (1 - ratio) * forget_img + (ratio) * retain_img
            
    elif mode.lower() == 'cutmix':
        mixed_img = forget_img.clone()
        bbx1, bby1, bbx2, bby2 = rand_bbox(retain_img.shape, 1 - ratio)
        mixed_img[:, :, bbx1:bbx2, bby1:bby2] = retain_img[:, :, bbx1:bbx2, bby1:bby2]
    else:
        raise ValueError(f"Unknown mixing mode: {mode}")
    return mixed_img


def apply_crop_resize(img, ratio=0.5, size=224):
    """Random crop and resize augmentation with configurable ratios"""
    min_size = max(int(ratio * size), 1)
    crop_size = min_size
    crop = transforms.RandomCrop(crop_size)
    resize = transforms.Resize(size)
    return resize(crop(img))


def apply_multi_crop_resize(img, num_crops=2, ratio=0.5, size=224):
    """Apply multiple random crops to the same image and resize them"""
    min_size = max(int(ratio * size), 1)
    crop_size = min_size
    crop = transforms.RandomCrop(crop_size)
    resize = transforms.Resize(size)
    
    crops = []
    for _ in range(num_crops):
        crops.append(resize(crop(img)))
    return crops


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
    
    seed = 2
    utils.setup_seed(seed)
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    model, retain_loader, forget_loader, val_loader = utils.setup_model_dataset(args)
    
    unlearn_data_loaders = OrderedDict(
        retain=retain_loader, forget=forget_loader, val=val_loader
    )
    
    # validate data choice
    data = args.data_type
    if data not in unlearn_data_loaders:
        raise ValueError(f"Invalid data type: {data}. Must be one of {list(unlearn_data_loaders.keys())}")
    
    original_model_path = args.model_path
    retrianed_model_path = args.retrained_model_path

    data_loader = unlearn_data_loaders[data]
    
    original_model = utils.load_model(original_model_path, device).to(device)
    retrained_model = utils.load_model(retrianed_model_path, device).to(device)
    original_model.eval()
    retrained_model.eval()
    
    global features_o, features_r
    
    # define augmentation methods
    augmentation_methods = {
        'original': lambda x: x,
        'gaussian': lambda x: add_gaussian_noise(x, sigma=0.1),
        'mixup': lambda x, y, ratio: mix_forget_retain(x, y, ratio=ratio, mode='mixup'),
        'cutmix': lambda x, y, ratio: mix_forget_retain(x, y, ratio=ratio, mode='cutmix'),
        'crop_resize': lambda x, ratio=0.5: apply_crop_resize(x, ratio=ratio),
        'multi_crop': lambda x, ratio=0.5: apply_multi_crop_resize(x, num_crops=2, ratio=ratio),
        'color_distortion': apply_color_distortion,
        'color_jitter': apply_color_jitter,
        'rotation': apply_rotation,
        'cutout': apply_cutout,
        'gaussian_blur': apply_gaussian_blur
    }

    method = args.aug_method[0]
    mix_ratios = np.arange(0.0, 1.1, 0.1)
    results = {}

    if method in ['mixup', 'cutmix', 'crop_resize', 'multi_crop']:
        mix_loader = unlearn_data_loaders["retain"] if data == "forget" else unlearn_data_loaders["forget"]
        
        for ratio in mix_ratios:
            print(f"\nTesting {method} ratio: {ratio:.1f}")
            batch_results = {
                'linear_cka': 0,
                'kernel_cka': 0,
                'linear_check': 0,
                'kernel_check': 0
            }
            
            mix_loader_iter = iter(mix_loader)
            
            for img, _ in tqdm(data_loader):
                img = img.to(device)
                
                if method in ['mixup', 'cutmix']:
                    try:
                        mix_image, _ = next(mix_loader_iter)
                    except StopIteration:
                        mix_loader_iter = iter(mix_loader)
                        mix_image, _ = next(mix_loader_iter)
                    
                    mix_image = mix_image.to(device)
                    aug_img = augmentation_methods[method](img, mix_image, ratio)
                elif method == 'multi_crop':
                    crops = augmentation_methods[method](img, ratio)
                    aug_img = crops[0]  # first crop
                    second_crop = crops[1]  # second crop
                    
                    features_o = []
                    features_r = []
                    
                    hooks_o = [original_model.avgpool.register_forward_hook(hook_fn_o)]
                    hooks_r = [retrained_model.avgpool.register_forward_hook(hook_fn_r)]
                    
                    with torch.no_grad():
                        # process first crop
                        original_model(aug_img)
                        retrained_model(second_crop)
                        f_o, f_r = features_o[0], features_r[0]
                        features_o.clear()
                        features_r.clear()
                        
                        # reshape features
                        f_o = f_o.view(f_o.size(0), -1)
                        f_r = f_r.view(f_r.size(0), -1)
                        
                        cuda_cka = CudaCKA(device)
                        # compare CKA between different crops
                        batch_results['linear_cka'] += cuda_cka.linear_CKA(f_o, f_r)
                        batch_results['kernel_cka'] += cuda_cka.kernel_CKA(f_o, f_r)
                        batch_results['linear_check'] += cuda_cka.linear_CKA(f_o, f_o)
                        batch_results['kernel_check'] += cuda_cka.kernel_CKA(f_r, f_r)
                else:  # original crop_resize
                    aug_img = augmentation_methods[method](img, ratio)
                
                features_o = []
                features_r = []
                
                hooks_o = [original_model.avgpool.register_forward_hook(hook_fn_o)]
                hooks_r = [retrained_model.avgpool.register_forward_hook(hook_fn_r)]
                
                with torch.no_grad():
                    original_model(aug_img)
                    retrained_model(aug_img)
                    
                    f_o, f_r = features_o[0], features_r[0]
                    f_o = f_o.view(f_o.size(0), -1)
                    f_r = f_r.view(f_r.size(0), -1)
                    
                    if method == 'mixup' or method == 'cutmix' or method == 'crop_resize':
                        cuda_cka = CudaCKA(device)
                        batch_results['linear_cka'] += cuda_cka.linear_CKA(f_o, f_r)
                        batch_results['kernel_cka'] += cuda_cka.kernel_CKA(f_o, f_r)
                        batch_results['linear_check'] += cuda_cka.linear_CKA(f_o, f_o)
                        batch_results['kernel_check'] += cuda_cka.kernel_CKA(f_r, f_r)
                
                for hook in hooks_o + hooks_r:
                    hook.remove()
                
                torch.cuda.empty_cache()
            results[ratio] = {k: v / len(data_loader) for k, v in batch_results.items()}
        
        # plot results for each ratio
        plot_results(results, method)
    
    else:
        # process non-mixing methods
        print(f"\nTesting {method} augmentation")
        batch_results = {
            'linear_cka': 0,
            'kernel_cka': 0,
            'linear_check': 0,
            'kernel_check': 0
        }
        
        for img, _ in tqdm(data_loader):
            img = img.to(device)
            aug_img = augmentation_methods[method](img)
            
            features_o = []
            features_r = []
            
            hooks_o = [original_model.avgpool.register_forward_hook(hook_fn_o)]
            hooks_r = [retrained_model.avgpool.register_forward_hook(hook_fn_r)]
            
            with torch.no_grad():
                original_model(aug_img)
                retrained_model(aug_img)
                
                f_o, f_r = features_o[0], features_r[0]
                f_o = f_o.view(f_o.size(0), -1)
                f_r = f_r.view(f_r.size(0), -1)
                
                cuda_cka = CudaCKA(device)
                batch_results['linear_cka'] += cuda_cka.linear_CKA(f_o, f_r)
                batch_results['kernel_cka'] += cuda_cka.kernel_CKA(f_o, f_r)
                batch_results['linear_check'] += cuda_cka.linear_CKA(f_o, f_o)
                batch_results['kernel_check'] += cuda_cka.kernel_CKA(f_r, f_r)
            
            for hook in hooks_o + hooks_r:
                hook.remove()
                
            torch.cuda.empty_cache()
        
        results[0] = {k: v / len(data_loader) for k, v in batch_results.items()}

    # print results
    print("\n=== Results across mix ratios ===")
    print("Ratio  Linear_CKA  Kernel_CKA  Linear_Check  Kernel_Check")
    print("-" * 55)
    
    for ratio in sorted(results.keys()):
        r = results[ratio]
        print(f"{ratio:4.1f}  {r['linear_cka']:10.4f}  {r['kernel_cka']:10.4f}  "
              f"{r['linear_check']:12.4f}  {r['kernel_check']:12.4f}")


def plot_results(results, method, save_dir='./plots'):
    """plot and save the CKA results as a bar chart"""
    os.makedirs(save_dir, exist_ok=True)
    
    ratios = sorted(results.keys())
    # convert GPU tensors to CPU and then to float values
    linear_cka_values = [float(results[r]['linear_cka'].cpu()) for r in ratios]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(ratios, linear_cka_values, width=0.08)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # customize the plot
    plt.xlabel('Ratio', fontsize=12)
    plt.ylabel('Linear CKA', fontsize=12)
    plt.title(f'Linear CKA vs Ratio for {method}', fontsize=14)
    
    # add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', rotation=0)
    
    # set y-axis to start from 0
    plt.ylim(0, max(linear_cka_values) * 1.1)
    
    # set x-axis ticks
    plt.xticks(ratios, [f'{r:.1f}' for r in ratios])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{method}_linear_cka.png'), dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
