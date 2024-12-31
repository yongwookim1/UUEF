import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision.models as models
import wandb

import utils
import main_cka
import main_knn
import arg_parser
from dataset import office_home_dataloaders,cub_dataloaders, domainnet126_dataloaders
from trainer import validate
from models import *


def evaluate_model(model_path, retrained_model, device, args):
    """evaluate model using kNN and CKA metrics on multiple datasets"""
    results = {}
    batch_size = 512
    
    (
        _,
        retain_loader,
        forget_loader,
        val_retain_loader,
        val_forget_loader
    ) = utils.setup_model_dataset(args)
    
    unlearn_data_loader = OrderedDict({
        "retain": retain_loader,
        "forget": forget_loader,
        "val_retain": val_retain_loader,
        "val_forget": val_forget_loader
    })
    
    # evaluate Accuracy
    print("Evaluating Accuracy...")
    model = utils.initialize_model(model_path, device)
    criterion = nn.CrossEntropyLoss()
    for name, data_loader in unlearn_data_loader.items():
        if name in ["val_retain", "val_forget"]:
            print(f"Validating {name} loader")
            val_acc = validate(data_loader, model, criterion, args)
            results[f"imagenet_{name}_acc"] = val_acc
            print(f"imagenet_{name}_acc: {val_acc}")
    
    # evaluate kNN
    print("Evaluating kNN...")
    knn_accuracy = utils.evaluate_knn(model_path, args)
    
    results.update({
        'imagenet_val_forget_knn': knn_accuracy['imagenet_val_forget'],
        'imagenet_val_retain_knn': knn_accuracy['imagenet_val_retain'],
        'office_home_real_world_knn': knn_accuracy['office_home_real_world'],
        'office_home_art_knn': knn_accuracy['office_home_art'],
        'office_home_clipart_knn': knn_accuracy['office_home_clipart'],
        'office_home_product_knn': knn_accuracy['office_home_product'],
        'cub_knn': knn_accuracy['cub'],
        'domainnet126_clipart_knn': knn_accuracy['domainnet126_clipart'],
        'domainnet126_painting_knn': knn_accuracy['domainnet126_painting'],
        'domainnet126_real_knn': knn_accuracy['domainnet126_real'],
        'domainnet126_sketch_knn': knn_accuracy['domainnet126_sketch'],
    })
    print(f"imagenet_val_forget_knn: {results['imagenet_val_forget_knn']}")
        
    # evaluate CKA
    print("Evaluating CKA...")
    model = utils.initialize_model(model_path, device)
    forget_cka = utils.evaluate_cka(model, retrained_model, forget_loader, device)
    retain_cka = utils.evaluate_cka(model, retrained_model, retain_loader, device)
    val_forget_cka = utils.evaluate_cka(model, retrained_model, val_forget_loader, device)
    val_retain_cka = utils.evaluate_cka(model, retrained_model, val_retain_loader, device)
    
    # Office-Home
    office_home_real_world_data_loader = utils.office_home_dataloaders(data_dir=args.office_home_dataset_path, domain="Real_World", batch_size=512, num_workers=4)
    office_home_art_data_loader = utils.office_home_dataloaders(data_dir=args.office_home_dataset_path, domain="Art", batch_size=512, num_workers=4)
    office_home_clipart_data_loader = utils.office_home_dataloaders(data_dir=args.office_home_dataset_path, domain="Clipart", batch_size=512, num_workers=4)
    office_home_product_data_loader = utils.office_home_dataloaders(data_dir=args.office_home_dataset_path, domain="Product", batch_size=512, num_workers=4)

    # CUB
    cub_data_loader = utils.cub_dataloaders(batch_size=512, data_dir=args.cub_dataset_path, num_workers=4)
    
    # DomainNet126
    domainnet126_clipart_data_loader = domainnet126_dataloaders(batch_size=512, domain='clipart', data_dir=args.domainnet_dataset_path, num_workers=4)
    domainnet126_painting_data_loader = domainnet126_dataloaders(batch_size=512, domain='painting', data_dir=args.domainnet_dataset_path, num_workers=4)
    domainnet126_real_data_loader = domainnet126_dataloaders(batch_size=512, domain='real', data_dir=args.domainnet_dataset_path, num_workers=4)
    domainnet126_sketch_data_loader = domainnet126_dataloaders(batch_size=512, domain='sketch', data_dir=args.domainnet_dataset_path, num_workers=4)

    model = utils.load_model(model_path, device).to(device)
    model.eval()
    
    office_home_real_world_results = utils.evaluate_cka(model, retrained_model, office_home_real_world_data_loader, device)
    office_home_art_results = utils.evaluate_cka(model, retrained_model, office_home_art_data_loader, device)
    office_home_clipart_results = utils.evaluate_cka(model, retrained_model, office_home_clipart_data_loader, device)
    office_home_product_results = utils.evaluate_cka(model, retrained_model, office_home_product_data_loader, device)
    
    cub_results = utils.evaluate_cka(model, retrained_model, cub_data_loader, device)
    
    domainnet126_clipart_results = utils.evaluate_cka(model, retrained_model, domainnet126_clipart_data_loader, device)
    domainnet126_painting_results = utils.evaluate_cka(model, retrained_model, domainnet126_painting_data_loader, device)
    domainnet126_real_results = utils.evaluate_cka(model, retrained_model, domainnet126_real_data_loader, device)
    domainnet126_sketch_results = utils.evaluate_cka(model, retrained_model, domainnet126_sketch_data_loader, device)
    
    
    results.update({
        "imagenet_forget_cka": forget_cka['cka'],
        "imagenet_retain_cka": retain_cka['cka'],
        "imagenet_val_forget_cka": val_forget_cka['cka'],
        "imagenet_val_retain_cka": val_retain_cka['cka'],
        "office_home_real_world_cka": office_home_real_world_results['cka'],
        "office_home_art_cka": office_home_art_results['cka'],
        "office_home_clipart_cka": office_home_clipart_results['cka'],
        "office_home_product_cka": office_home_product_results['cka'],
        "cub_cka": cub_results['cka'],
        "domainnet126_clipart_cka": domainnet126_clipart_results['cka'],
        "domainnet126_painting_cka": domainnet126_painting_results['cka'],
        "domainnet126_real_cka": domainnet126_real_results['cka'],
        "domainnet126_sketch_cka": domainnet126_sketch_results['cka']
    })

    return results


def main():
    args = arg_parser.parse_args()
    utils.setup_seed(2)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # initialize wandb
    if args.use_wandb:
        run = utils.init_wandb(args)
        
    model_paths = [
        # "./pretrained_model/0model_SA_best159.pth.tar",
        # "./pretrained_model/retraincheckpoint100.pth.tar",
        "./result/GA/GA/5e-06/5/GAcheckpoint.pth.tar",
        "./result/RL/RL_imagenet/5e-06/2/RL_imagenetcheckpoint.pth.tar",
        "./result/SalUn/RL_imagenet/5e-06/2/RL_imagenetcheckpoint.pth.tar",
        "./result/CU/CU/0.001/79/CUcheckpoint.pth.tar",
        "./result/SCAR/SCAR/0.0005/19/SCARcheckpoint.pth.tar",
        "./result/SCRUB/SCRUB/1e-05/90/SCRUBcheckpoint.pth.tar",
        "./result/GAwithKD/GAwithKD/1e-05/20/GAwithKDcheckpoint.pth.tar",
        "./result/RKD/RKD/7e-06/14/RKDcheckpoint.pth.tar",
        "./result/AKD/AKD/7e-06/14/AKDcheckpoint.pth.tar",
        "./result/SPKD/SPKD/7e-06/14/SPKDcheckpoint.pth.tar",
    ]

    retrained_model = utils.load_model(args.retrained_model_path, device)
    
    results = {}
    
    for model_path in model_paths:
        eval_results = evaluate_model(model_path, retrained_model, device, args)
        results.update(eval_results)
        
        # print results
        print("Evaluation results:")
        print("-" * 50)
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
        
        method_name = model_path.split('/')[-3].replace('checkpoint','')
        if args.use_wandb:
            wandb.log({f"{method_name}/{k}": v for k, v in results.items()})
        
        # save results
        save_dir = f"result/evaluation/{method_name}"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "evaluation_results.pt")
        torch.save(results, save_path)
        print(f"Results saved to {save_path}")


if __name__ == "__main__":
    main()