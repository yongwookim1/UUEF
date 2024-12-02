from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
import os
import random

import arg_parser
import utils
from CKA import CudaCKA

def replace_loader_dataset(dataset, batch_size, seed=1, shuffle=False):
    utils.setup_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )

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

def main():
    data = "retain"
    args = arg_parser.parse_args()
    seed = 2
    utils.setup_seed(seed)
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    model, retain_loader, forget_loader, val_loader = utils.setup_model_dataset(args)
    
    forget_loader = replace_loader_dataset(
        forget_loader.dataset, batch_size=args.batch_size, seed=seed, shuffle=False
    )
    retain_loader = replace_loader_dataset(
        retain_loader.dataset, batch_size=args.batch_size, seed=seed, shuffle=False
    )
    
    unlearn_data_loaders = OrderedDict(
        retain=retain_loader, forget=forget_loader, val=val_loader
    )
    
    pretrained_model_paths = [
        "/home/kyw1654/unlearning/baseline/pretrained_model/0model_SA_best159.pth.tar",
        "/home/kyw1654/unlearning/baseline/pretrained_model/retraincheckpoint100.pth.tar",
    ]
    
    data_loader = unlearn_data_loaders[data]
    
    original_model = load_model(pretrained_model_paths[0], device).to(device)
    retrained_model = load_model(pretrained_model_paths[1], device).to(device)
    
    original_model.eval()
    retrained_model.eval()
    
    global features_o, features_r
    linear_cka = kernel_cka = linear_check = kernel_check = 0
    
    for i, data in enumerate(tqdm(data_loader)):
        img, _ = data
        img = img.to(device)
        
        features_o = []
        features_r = []
        
        hooks_o = []
        hooks_r = []
        
        hooks_o.append(original_model.avgpool.register_forward_hook(hook_fn_o))
        hooks_r.append(retrained_model.avgpool.register_forward_hook(hook_fn_r))
        
        with torch.no_grad():
            original_output = original_model(img)
            retrained_output = retrained_model(img)
            
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
    
    print(f"Linear CKA: {linear_cka / len(data_loader):.4f}")
    print(f"Kernel CKA: {kernel_cka / len(data_loader):.4f}")
    print(f"Linear CKA check: {linear_check / len(data_loader):.4f}")
    print(f"Kernel CKA check: {kernel_check / len(data_loader):.4f}")

if __name__ == "__main__":
    main()
