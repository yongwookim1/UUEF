import os
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import wandb

import utils
import main_cka
import main_knn
import arg_parser
from dataset import office_home_dataloaders,cub_dataloaders, domainnet126_dataloaders
from trainer import validate
from models import *
import evaluation


def replace_loader_dataset(
        dataset, batch_size=512, seed=2, shuffle=False
    ):
        utils.setup_seed(seed)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
            shuffle=shuffle,
        )


def evaluate_model(model_path, retrained_model_path, device, args):
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
    
    forget_loader = replace_loader_dataset(forget_loader.dataset, batch_size=512, seed=1, shuffle=False)
    retain_loader = replace_loader_dataset(retain_loader.dataset, batch_size=512, seed=1, shuffle=False)
    val_forget_loader = replace_loader_dataset(val_forget_loader.dataset, batch_size=512, seed=1, shuffle=False)
    val_retain_loader = replace_loader_dataset(val_retain_loader.dataset, batch_size=512, seed=1, shuffle=False)
    
    # evaluate Accuracy
    print("Evaluating Accuracy...")
    model = utils.initialize_model(model_path, device)
    criterion = nn.CrossEntropyLoss()
    for name, data_loader in unlearn_data_loader.items():
        print(f"Validating {name} loader")
        val_acc = validate(data_loader, model, criterion, args)
        results[f"imagenet_{name}_acc"] = val_acc
        print(f"imagenet_{name}_acc: {val_acc}")
    
    # evaluate kNN
    print("Evaluating kNN...")
    knn_accuracy = utils.evaluate_knn(model_path, args)
    
    results.update({
        'imagenet_forget_knn': knn_accuracy['imagenet_forget'],
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
        
    # evaluate CKA
    print("Evaluating CKA...")
    model = utils.initialize_model(model_path, device)
    retrained_model = utils.initialize_model(retrained_model_path, device)
    forget_cka = utils.evaluate_cka(model, retrained_model, forget_loader, device, args=args, data='Df')
    retain_cka = utils.evaluate_cka(model, retrained_model, retain_loader, device, args=args, data='Dr')
    val_forget_cka = utils.evaluate_cka(model, retrained_model, val_forget_loader, device, args=args, data='Dtf')
    val_retain_cka = utils.evaluate_cka(model, retrained_model, val_retain_loader, device, args=args, data='Dtr')
    
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
    retrained_model = utils.load_model(retrained_model_path, device).to(device)
    model.eval()
    retrained_model.eval()
    
    office_home_real_world_results = utils.evaluate_cka(model, retrained_model, office_home_real_world_data_loader, device, args=args)
    office_home_art_results = utils.evaluate_cka(model, retrained_model, office_home_art_data_loader, device, args=args)
    office_home_clipart_results = utils.evaluate_cka(model, retrained_model, office_home_clipart_data_loader, device, args=args)
    office_home_product_results = utils.evaluate_cka(model, retrained_model, office_home_product_data_loader, device, args=args)
    
    cub_results = utils.evaluate_cka(model, retrained_model, cub_data_loader, device, args=args)
    
    domainnet126_clipart_results = utils.evaluate_cka(model, retrained_model, domainnet126_clipart_data_loader, device, args=args)
    domainnet126_painting_results = utils.evaluate_cka(model, retrained_model, domainnet126_painting_data_loader, device, args=args)
    domainnet126_real_results = utils.evaluate_cka(model, retrained_model, domainnet126_real_data_loader, device, args=args)
    domainnet126_sketch_results = utils.evaluate_cka(model, retrained_model, domainnet126_sketch_data_loader, device, args=args)
    
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
        "domainnet126_sketch_cka": domainnet126_sketch_results['cka'],
        })
    
    # MIA
    # model = utils.initialize_model(model_path, device)
    
    # forget_dataset = forget_loader.dataset
    # retain_dataset = retain_loader.dataset
    # forget_len = len(forget_dataset)
    # retain_len = len(retain_dataset)

    # utils.dataset_convert_to_test(retain_dataset, args)
    # utils.dataset_convert_to_test(forget_loader, args)

    # shadow_train = torch.utils.data.Subset(retain_dataset, list(range(1000)))
    # shadow_train_loader = torch.utils.data.DataLoader(
    #     shadow_train, batch_size=args.batch_size, shuffle=False
    # )
    
    # val_forget_dataset = val_forget_loader.dataset
    # val_retain_dataset = val_retain_loader.dataset
    # test_dataset = torch.utils.data.ConcatDataset([val_forget_dataset, val_retain_dataset])
    
    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset, batch_size=args.batch_size, shuffle=False
    # )
    
    # evaluation_result = evaluation.SVC_MIA(
    #     shadow_train=shadow_train_loader,
    #     shadow_test=test_loader,
    #     target_train=None,
    #     target_test=forget_loader,
    #     model=model,
    # )
    
    # results.update({
    #     "SVC_MIA_forget_correctness": evaluation_result['correctness'],
    #     "SVC_MIA_forget_prob": evaluation_result['prob'],
    # })

    return results


def main():
    args = arg_parser.parse_args()
    utils.setup_seed(2)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # initialize wandb
    if args.use_wandb:
        run = utils.init_wandb(args, project_name="unlearning_evaluation")
        
    model_paths = [
        "./pretrained_model/original.pth.tar",
        # "./pretrained_model/retrained_random100.pth.tar",
        # "./result/GA/GA/5e-06/7/GAcheckpoint.pth.tar",
        # "./result/RL/RL_imagenet/5e-06/6/RL_imagenetcheckpoint.pth.tar",
        # "./result/SalUn/RL_imagenet/5e-06/9/RL_imagenetcheckpoint.pth.tar",
        # "./result/FT/FT/0.0001/39/FTcheckpoint.pth.tar",
        # "./result/CU/CU/0.001/79/CUcheckpoint.pth.tar",
        # "./result/SCAR/SCAR/0.0005/26/SCARcheckpoint.pth.tar",
        # "./result/SCRUB/SCRUB/1e-05/90/SCRUBcheckpoint.pth.tar",
        # "./result/GAwithKD/GAwithKD/1e-05/20/GAwithKDcheckpoint.pth.tar",
        # "./result/RKD/RKD/7e-06/14/RKDcheckpoint.pth.tar",
        # "./result/AKD/AKD/7e-06/14/AKDcheckpoint.pth.tar",
        # "./result/SPKD/SPKD/7e-06/14/SPKDcheckpoint.pth.tar",
        # "./result/SPKD_retrained/SPKD_retrained/7e-06/14/SPKD_retrainedcheckpoint.pth.tar",
        # "./pretrained_model/retrained_top100_officehome.pth.tar",
        # "./result/GA_top100_officehome/GA/5e-06/7/GAcheckpoint.pth.tar",
        # "./result/RL_top100_officehome/RL_imagenet/5e-06/6/RL_imagenetcheckpoint.pth.tar",
        # "./result/SalUn_top100_officehome/RL_imagenet/5e-06/9/RL_imagenetcheckpoint.pth.tar",
        # "./result/FT_top100_officehome/FT/0.0001/39/FTcheckpoint.pth.tar",
        # "./result/CU_top100_officehome/CU/0.001/79/CUcheckpoint.pth.tar",
        # "./result/SCAR_top100_officehome/SCAR/0.0005/26/SCARcheckpoint.pth.tar",
        # "./result/SCRUB_top100_officehome/SCRUB/1e-05/90/SCRUBcheckpoint.pth.tar",
        # "./result/GA_KD_top100_officehome/GA_KD/1e-05/20/GA_KDcheckpoint.pth.tar",
    ]
    
    eval_results = evaluate_model(args.model_path, args.retrained_model_path, device, args)
    results.update(eval_results)
    
    # print results
    print("Evaluation results:")
    print("-" * 50)
    for metric, value in results.items():
        print(f"{metric}: {value:.2f}")

    try:
        method_name = args.model_path.split('/')[-4]
    except:
        method_name = "original" if "original" in args.model_path else "retrained"
        
    if args.use_wandb:
        wandb_results = {}
        for metric_name, value in results.items():
            wandb_results[f"{metric_name}"] = value
        
        wandb.log(wandb_results)


if __name__ == "__main__":
    main()