import copy
import os
from collections import OrderedDict

import arg_parser
import evaluation
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import wandb
import unlearn
import utils
from trainer import validate
import main_cka
import main_knn


def main():
    args = arg_parser.parse_args()
    
    # initialize wandb if enabled
    if args.use_wandb:
        run = utils.init_wandb(args)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        utils.setup_seed(args.seed)
    seed = args.seed
    # prepare dataset
    if args.dataset == 'imagenet' :
        (
            model,
            retain_loader,
            forget_loader,
            val_retain_loader,
            val_forget_loader
        ) = utils.setup_model_dataset(args)
    else:
        (
            model,
            train_loader_full,
            val_loader,
            test_loader,
            marked_loader,
        ) = utils.setup_model_dataset(args)
    model.to(device)


    def replace_loader_dataset(
        dataset, batch_size=args.batch_size, seed=1, shuffle=True
    ):
        utils.setup_seed(seed)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
            shuffle=shuffle,
        )
    if args.dataset == 'imagenet' :
        forget_dataset = copy.deepcopy(forget_loader.dataset)
    else:
        forget_dataset = copy.deepcopy(marked_loader.dataset)
    if args.dataset == "svhn":
        try:
            marked = forget_dataset.targets < 0
        except:
            marked = forget_dataset.labels < 0
        forget_dataset.data = forget_dataset.data[marked]
        try:
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
        except:
            forget_dataset.labels = -forget_dataset.labels[marked] - 1
        forget_loader = replace_loader_dataset(forget_dataset, seed=seed, shuffle=True)
        retain_dataset = copy.deepcopy(marked_loader.dataset)
        try:
            marked = retain_dataset.targets >= 0
        except:
            marked = retain_dataset.labels >= 0
        retain_dataset.data = retain_dataset.data[marked]
        try:
            retain_dataset.targets = retain_dataset.targets[marked]
        except:
            retain_dataset.labels = retain_dataset.labels[marked]
        retain_loader = replace_loader_dataset(retain_dataset, seed=seed, shuffle=True)
        assert len(forget_dataset) + len(retain_dataset) == len(
            train_loader_full.dataset
        )
    elif args.dataset == 'imagenet' :
        forget_loader = replace_loader_dataset(
                forget_loader.dataset, seed=seed, shuffle=True
        )
        retain_loader = replace_loader_dataset(
                retain_loader.dataset, seed=seed, shuffle=True
        )
    else:
        try:
            marked = forget_dataset.targets < 0
            forget_dataset.data = forget_dataset.data[marked]
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
            forget_loader = replace_loader_dataset(
                forget_dataset, seed=seed, shuffle=True
            )
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            marked = retain_dataset.targets >= 0
            retain_dataset.data = retain_dataset.data[marked]
            retain_dataset.targets = retain_dataset.targets[marked]
            retain_loader = replace_loader_dataset(
                retain_dataset, seed=seed, shuffle=True
            )
            assert len(forget_dataset) + len(retain_dataset) == len(
                train_loader_full.dataset
            )
        except:
            marked = forget_dataset.targets < 0
            forget_dataset.imgs = forget_dataset.imgs[marked]
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
            forget_loader = replace_loader_dataset(
                forget_dataset, seed=seed, shuffle=True
            )
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            marked = retain_dataset.targets >= 0
            retain_dataset.imgs = retain_dataset.imgs[marked]
            retain_dataset.targets = retain_dataset.targets[marked]
            retain_loader = replace_loader_dataset(
                retain_dataset, seed=seed, shuffle=True
            )
            assert len(forget_dataset) + len(retain_dataset) == len(
                train_loader_full.dataset
            )

    print(f"number of retain dataset {len(retain_loader.dataset)}")
    print(f"number of forget dataset {len(forget_loader.dataset)}")
    if args.dataset == 'imagenet' :
        unlearn_data_loaders = OrderedDict(
            retain=retain_loader, forget=forget_loader, val_retain=val_retain_loader, val_forget=val_forget_loader
        )
    else :
        unlearn_data_loaders = OrderedDict(
            retain=retain_loader, forget=forget_loader, val=val_loader, test=test_loader
        )

    criterion = nn.CrossEntropyLoss()
    evaluation_result = None

    if args.resume:
        checkpoint = unlearn.load_unlearn_checkpoint(model, device, args)

    if args.resume and checkpoint is not None:
        model, evaluation_result = checkpoint
    else:
        checkpoint = torch.load(args.model_path, map_location=device)
        if "state_dict" in checkpoint.keys():
            checkpoint = checkpoint["state_dict"]
        
        checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        
        if args.unlearn != "retrain":
            model.load_state_dict(checkpoint, strict=True)

        unlearn_method = unlearn.get_unlearn_method(args.unlearn)
        unlearn_method(unlearn_data_loaders, model, criterion, args)
        
        accuracy = {}
        for name, loader in unlearn_data_loaders.items():
            print(f"Validating {name} loader")
            val_acc = validate(loader, model, criterion, args)
            accuracy[name] = val_acc
            print(f"{name} acc: {val_acc}")
        metrics = {
        "epoch": args.unlearn_epochs,
        f"{args.dataset}_retain_acc": accuracy["retain"],
        f"{args.dataset}_forget_acc": accuracy["forget"],
        f"{args.dataset}_val_retain_acc": accuracy["val_retain"],
        f"{args.dataset}_val_forget_acc": accuracy["val_forget"],
        }
        
        knn_results = main_knn.main()
        
        metrics = {
            f"office_home_real_world_knn": knn_results["office_home_real_world"],
            f"office_home_art_knn": knn_results["office_home_art"],
            f"office_home_clipart_knn": knn_results["office_home_clipart"],
            f"office_home_product_knn": knn_results["office_home_product"],
            f"cub_knn": knn_results["cub"],
            f"domainnet126_clipart_knn": knn_results["domainnet126_clipart"],
            f"domainnet126_painting_knn": knn_results["domainnet126_painting"],
            f"domainnet126_real_knn": knn_results["domainnet126_real"],
            f"domainnet126_sketch_knn": knn_results["domainnet126_sketch"],
        }
        
        cka_results = main_cka.main()
        
        metrics = {
            f"office_home_real_world_cka": cka_results["office_home_real_world_cka"],
            f"office_home_art_cka": cka_results["office_home_art_cka"],
            f"office_home_clipart_cka": cka_results["office_home_clipart_cka"],
            f"office_home_product_cka": cka_results["office_home_product_cka"],
            f"cub_cka": cka_results["cub_cka"],
            f"domainnet126_clipart_cka": cka_results["domainnet126_clipart_cka"],
            f"domainnet126_painting_cka": cka_results["domainnet126_painting_cka"],
            f"domainnet126_real_cka": cka_results["domainnet126_real_cka"],
            f"domainnet126_sketch_cka": cka_results["domainnet126_sketch_cka"],
        }
        
        wandb.log(metrics)

        unlearn.save_unlearn_checkpoint(model, None, args)
    
    wandb.finish()

    # if evaluation_result is None:
    #     evaluation_result = {}

    # if "new_accuracy" not in evaluation_result:
    #     accuracy = {}
    #     for name, loader in unlearn_data_loaders.items():
    #         val_acc = validate(loader, model, criterion, args)
    #         accuracy[name] = val_acc
    #         print(f"{name} acc: {val_acc}")

    #     evaluation_result["accuracy"] = accuracy
    #     unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

    # for deprecated in ["MIA", "SVC_MIA", "SVC_MIA_forget"]:
    #     if deprecated in evaluation_result:
    #         evaluation_result.pop(deprecated)

    # """forget efficacy MIA:
    #     in distribution: retain
    #     out of distribution: test
    #     target: (, forget)"""
    # if "SVC_MIA_forget_efficacy" not in evaluation_result:
    #     forget_len = len(forget_dataset)
    #     retain_len = len(retain_dataset)

    #     utils.dataset_convert_to_test(retain_dataset, args)
    #     utils.dataset_convert_to_test(forget_loader, args)

    #     shadow_train = torch.utils.data.Subset(retain_dataset, list(range(1000)))
    #     shadow_train_loader = torch.utils.data.DataLoader(
    #         shadow_train, batch_size=args.batch_size, shuffle=False
    #     )

    #     evaluation_result["SVC_MIA_forget_efficacy"] = evaluation.SVC_MIA(
    #         shadow_train=shadow_train_loader,
    #         shadow_test=test_loader,
    #         target_train=None,
    #         target_test=forget_loader,
    #         model=model,
    #     )
    #     unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

    # unlearn.save_unlearn_checkpoint(model, evaluation_result, args)


if __name__ == "__main__":
    main()
