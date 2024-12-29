import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm
import os
from PIL import Image
import matplotlib.pyplot as plt

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
    state_dict = {k: v for k, v in state_dict.items() if not (k.startswith('normalize.'))}
    model.load_state_dict(state_dict, strict=True)
    print(f"Model loading complete: {pretrained_model_path}")
    return model


class FeatureExtractor:
    def __init__(self):
        self.features = []
        
    def hook_fn(self, module, input, output):
        self.features.append(output)
        
    def clear(self):
        self.features = []


def plot_cka_results(cka_values, layer_names, save_path='./plots/cka_results.png'):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(cka_values)), cka_values)
    
    plt.xlabel('Layers')
    plt.ylabel('Linear CKA')
    plt.title('Original model and Retrained model CKA Values Across Different Layers')
    plt.xticks(range(len(layer_names)), layer_names, rotation=45)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def evaluate_cka(model, retrained_model, data_loader, device, mode='all'):
    """
    compute CKA similarity between two models
    
    args:
        model: first model to compare
        retrained_model: second model to compare
        data_loader: dataLoader containing the dataset
        device: torch device to use
        mode: 'all' for all layers or 'avgpool' for only avgpool layer
    """
    model.eval()
    retrained_model.eval()

    # feature extractors
    original_extractor = FeatureExtractor()
    retrained_extractor = FeatureExtractor()
    cuda_cka = CudaCKA(device)

    if mode == 'all':
        layers = ['layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc']
        cka_result = {f"linear_cka{i}": 0 for i in range(len(layers))}
        cka_result.update({f"linear_check{i}": 0 for i in range(len(layers))})
    else:
        linear_cka = kernel_cka = linear_check = kernel_check = 0
        cuda_cka = CudaCKA(device)

    for data in tqdm(data_loader):
        img, _ = data
        img = img.to(device)

        if mode == 'all':
            hooks = [
                getattr(model, layer).register_forward_hook(original_extractor.hook_fn)
                for layer in layers
            ] + [
                getattr(retrained_model, layer).register_forward_hook(retrained_extractor.hook_fn)
                for layer in layers
            ]
        else:
            hooks = [
                model.avgpool.register_forward_hook(original_extractor.hook_fn),
                retrained_model.avgpool.register_forward_hook(retrained_extractor.hook_fn)
            ]

        with torch.no_grad():
            model(img)
            retrained_model(img)

            if mode == 'all':
                for i in range(len(layers)):
                    f_o = original_extractor.features[i].view(original_extractor.features[i].size(0), -1)
                    f_r = retrained_extractor.features[i].view(retrained_extractor.features[i].size(0), -1)
                    cka_result[f"linear_cka{i}"] += cuda_cka.linear_CKA(f_o, f_r)
                    cka_result[f"linear_check{i}"] += cuda_cka.linear_CKA(f_o, f_o)
            else:
                f_o = original_extractor.features[0].view(original_extractor.features[0].size(0), -1)
                f_r = retrained_extractor.features[0].view(retrained_extractor.features[0].size(0), -1)
                linear_cka += cuda_cka.linear_CKA(f_o, f_r)
                kernel_cka += cuda_cka.kernel_CKA(f_o, f_r)
                linear_check += cuda_cka.linear_CKA(f_o, f_o)
                kernel_check += cuda_cka.kernel_CKA(f_r, f_r)

        for hook in hooks:
            hook.remove()
        original_extractor.clear()
        retrained_extractor.clear()

    n = len(data_loader)
    if mode == 'all':
        final_results = {}
        cka_values = []
        for i, layer in enumerate(layers):
            avg_cka = cka_result[f"linear_cka{i}"] / n
            avg_check = cka_result[f"linear_check{i}"] / n
            cka_values.append(avg_cka)
            final_results[layer] = {
                'cka': avg_cka.cpu().numpy(),
            }
        
        return final_results
    else:
        return {
            'cka': linear_cka / n,
        }
        
        
def evaluate_all_cka(model):
    args = arg_parser.parse_args()
    utils.setup_seed(2)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # Office-Home
    office_home_real_world_data_loader = utils.office_home_dataloaders(data_dir=args.office_home_dataset_path, domain="Real_World", batch_size=512, num_workers=4)
    office_home_art_data_loader = utils.office_home_dataloaders(data_dir=args.office_home_dataset_path, domain="Art", batch_size=512, num_workers=4)
    office_home_clipart_data_loader = utils.office_home_dataloaders(data_dir=args.office_home_dataset_path, domain="Clipart", batch_size=512, num_workers=4)
    office_home_product_data_loader = utils.office_home_dataloaders(data_dir=args.office_home_dataset_path, domain="Product", batch_size=512, num_workers=4)

    # CUB
    cub_data_loader = utils.cub_dataloaders(batch_size=512, data_dir=args.cub_dataset_path, num_workers=4)
    
    # DomainNet126
    domainnet126_clipart_data_loader = utils.domainnet126_dataloaders(batch_size=512, domain='clipart', data_dir=args.domainnet_dataset_path, num_workers=4)
    domainnet126_painting_data_loader = utils.domainnet126_dataloaders(batch_size=512, domain='painting', data_dir=args.domainnet_dataset_path, num_workers=4)
    domainnet126_real_data_loader = utils.domainnet126_dataloaders(batch_size=512, domain='real', data_dir=args.domainnet_dataset_path, num_workers=4)
    domainnet126_sketch_data_loader = utils.domainnet126_dataloaders(batch_size=512, domain='sketch', data_dir=args.domainnet_dataset_path, num_workers=4)
    
    model = model.to(device)
    retrained_model = load_model(args.retrained_model_path, device).to(device)

    # add mode selection based on args if needed
    mode = 'avgpool'
    office_home_real_world_results = evaluate_cka(model, retrained_model, office_home_real_world_data_loader, device, mode=mode)
    office_home_art_results = evaluate_cka(model, retrained_model, office_home_art_data_loader, device, mode=mode)
    office_home_clipart_results = evaluate_cka(model, retrained_model, office_home_clipart_data_loader, device, mode=mode)
    office_home_product_results = evaluate_cka(model, retrained_model, office_home_product_data_loader, device, mode=mode)
    
    cub_results = evaluate_cka(model, retrained_model, cub_data_loader, device, mode=mode)
    
    domainnet126_clipart_results = evaluate_cka(model, retrained_model, domainnet126_clipart_data_loader, device, mode=mode)
    domainnet126_painting_results = evaluate_cka(model, retrained_model, domainnet126_painting_data_loader, device, mode=mode)
    domainnet126_real_results = evaluate_cka(model, retrained_model, domainnet126_real_data_loader, device, mode=mode)
    domainnet126_sketch_results = evaluate_cka(model, retrained_model, domainnet126_sketch_data_loader, device, mode=mode)

    print(f"Office-Home Real World CKA: {office_home_real_world_results}")
    print(f"Office-Home Art CKA: {office_home_art_results}")
    print(f"Office-Home Clipart CKA: {office_home_clipart_results}")
    print(f"Office-Home Product CKA: {office_home_product_results}")
    print(f"CUB CKA: {cub_results}")
    print(f"DomainNet-clipart CKA: {domainnet126_clipart_results}")
    print(f"DomainNet-painting CKA: {domainnet126_painting_results}")
    print(f"DomainNet-real CKA: {domainnet126_real_results}")
    print(f"DomainNet-sketch CKA: {domainnet126_sketch_results}")
    
    return {
        "office_home_real_world_cka": office_home_real_world_results,
        "office_home_art_cka": office_home_art_results,
        "office_home_clipart_cka": office_home_clipart_results,
        "office_home_product_cka": office_home_product_results,
        "cub_cka": cub_results,
        "domainnet126_clipart_cka": domainnet126_clipart_results,
        "domainnet126_painting_cka": domainnet126_painting_results,
        "domainnet126_real_cka": domainnet126_real_results,
        "domainnet126_sketch_cka": domainnet126_sketch_results,
    }


def main():
    args = arg_parser.parse_args()
    utils.setup_seed(2)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # Office-Home
    office_home_real_world_data_loader = utils.office_home_dataloaders(data_dir=args.office_home_dataset_path, domain="Real_World", batch_size=512, num_workers=4)
    office_home_art_data_loader = utils.office_home_dataloaders(data_dir=args.office_home_dataset_path, domain="Art", batch_size=512, num_workers=4)
    office_home_clipart_data_loader = utils.office_home_dataloaders(data_dir=args.office_home_dataset_path, domain="Clipart", batch_size=512, num_workers=4)
    office_home_product_data_loader = utils.office_home_dataloaders(data_dir=args.office_home_dataset_path, domain="Product", batch_size=512, num_workers=4)

    # CUB
    cub_data_loader = utils.cub_dataloaders(batch_size=512, data_dir=args.cub_dataset_path, num_workers=4)
    
    # DomainNet126
    domainnet126_clipart_data_loader = utils.domainnet126_dataloaders(batch_size=512, domain='clipart', data_dir=args.domainnet_dataset_path, num_workers=4)
    domainnet126_painting_data_loader = utils.domainnet126_dataloaders(batch_size=512, domain='painting', data_dir=args.domainnet_dataset_path, num_workers=4)
    domainnet126_real_data_loader = utils.domainnet126_dataloaders(batch_size=512, domain='real', data_dir=args.domainnet_dataset_path, num_workers=4)
    domainnet126_sketch_data_loader = utils.domainnet126_dataloaders(batch_size=512, domain='sketch', data_dir=args.domainnet_dataset_path, num_workers=4)
    
    model = load_model(args.model_path, device).to(device)
    retrained_model = load_model(args.retrained_model_path, device).to(device)

    # add mode selection based on args if needed
    mode = 'avgpool'
    office_home_real_world_results = evaluate_cka(model, retrained_model, office_home_real_world_data_loader, device, mode=mode)
    office_home_art_results = evaluate_cka(model, retrained_model, office_home_art_data_loader, device, mode=mode)
    office_home_clipart_results = evaluate_cka(model, retrained_model, office_home_clipart_data_loader, device, mode=mode)
    office_home_product_results = evaluate_cka(model, retrained_model, office_home_product_data_loader, device, mode=mode)
    
    cub_results = evaluate_cka(model, retrained_model, cub_data_loader, device, mode=mode)
    
    domainnet126_clipart_results = evaluate_cka(model, retrained_model, domainnet126_clipart_data_loader, device, mode=mode)
    domainnet126_painting_results = evaluate_cka(model, retrained_model, domainnet126_painting_data_loader, device, mode=mode)
    domainnet126_real_results = evaluate_cka(model, retrained_model, domainnet126_real_data_loader, device, mode=mode)
    domainnet126_sketch_results = evaluate_cka(model, retrained_model, domainnet126_sketch_data_loader, device, mode=mode)

    print(f"Office-Home Real World CKA: {office_home_real_world_results}")
    print(f"Office-Home Art CKA: {office_home_art_results}")
    print(f"Office-Home Clipart CKA: {office_home_clipart_results}")
    print(f"Office-Home Product CKA: {office_home_product_results}")
    print(f"CUB CKA: {cub_results}")
    print(f"DomainNet-clipart CKA: {domainnet126_clipart_results}")
    print(f"DomainNet-painting CKA: {domainnet126_painting_results}")
    print(f"DomainNet-real CKA: {domainnet126_real_results}")
    print(f"DomainNet-sketch CKA: {domainnet126_sketch_results}")
    
    return {
        "office_home_real_world_cka": office_home_real_world_results,
        "office_home_art_cka": office_home_art_results,
        "office_home_clipart_cka": office_home_clipart_results,
        "office_home_product_cka": office_home_product_results,
        "cub_cka": cub_results,
        "domainnet126_clipart_cka": domainnet126_clipart_results,
        "domainnet126_painting_cka": domainnet126_painting_results,
        "domainnet126_real_cka": domainnet126_real_results,
        "domainnet126_sketch_cka": domainnet126_sketch_results,
    }


if __name__ == "__main__":
    main()
