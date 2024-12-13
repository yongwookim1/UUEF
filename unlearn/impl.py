import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import utils
import wandb
from torch.utils.data import DataLoader

import main_knn
from main_cka import OfficeHomeDataset
from trainer.val import validate
from pruner import extract_mask, prune_model_custom, remove_prune
import pruner

def plot_training_curve(training_result, save_dir, prefix):
    # plot training curve
    for name, result in training_result.items():
        plt.plot(result, label=f"{name}_acc")
    plt.legend()
    plt.savefig(os.path.join(save_dir, prefix + "_train.png"))
    plt.close()


def save_unlearn_checkpoint(model, evaluation_result, args):
    state = {"state_dict": model.state_dict(), "evaluation_result": evaluation_result}
    utils.save_checkpoint(state, False, args.save_dir, args.unlearn)
    utils.save_checkpoint(
        evaluation_result,
        False,
        args.save_dir,
        args.unlearn,
        filename="eval_result.pth.tar",
    )


def load_unlearn_checkpoint(model, device, args):
    checkpoint = utils.load_checkpoint(device, args.save_dir, args.unlearn)
    if checkpoint is None or checkpoint.get("state_dict") is None:
        return None

    current_mask = pruner.extract_mask(checkpoint["state_dict"])
    pruner.prune_model_custom(model, current_mask)
    pruner.check_sparsity(model)

    model.load_state_dict(checkpoint["state_dict"])

    # adding an extra forward process to enable the masks
    x_rand = torch.rand(1, 3, args.input_size, args.input_size).to(device)
    model.eval()
    with torch.no_grad():
        model(x_rand)

    evaluation_result = checkpoint.get("evaluation_result")
    return model, evaluation_result


def _iterative_unlearn_impl(unlearn_iter_func):
    def _wrapped(data_loaders, model, criterion, args, mask=None, **kwargs):
        decreasing_lr = list(map(int, args.decreasing_lr.split(",")))
        if args.rewind_epoch != 0:
            initialization = torch.load(
                args.rewind_pth, map_location=torch.device("cuda:" + str(args.gpu))
            )
            current_mask = extract_mask(model.state_dict())
            remove_prune(model)
            # weight rewinding
            # rewind, initialization is a full model architecture without masks
            model.load_state_dict(initialization, strict=True)
            prune_model_custom(model, current_mask)
    
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.unlearn_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

        if args.imagenet_arch and args.unlearn == "retrain":
            lambda0 = (
                lambda cur_iter: (cur_iter + 1) / args.warmup
                if cur_iter < args.warmup
                else (
                    0.5
                    * (
                        1.0
                        + np.cos(
                            np.pi
                            * (
                                (cur_iter - args.warmup)
                                / (args.unlearn_epochs - args.warmup)
                            )
                        )
                    )
                )
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=decreasing_lr, gamma=0.1
            )  # 0.1 is fixed
        if args.rewind_epoch != 0:
            # learning rate rewinding
            for _ in range(args.rewind_epoch):
                scheduler.step()
        for epoch in range(0, args.unlearn_epochs):
            start_time = time.time()

            print(
                "Epoch #{}, Learning rate: {}".format(
                    epoch, optimizer.state_dict()["param_groups"][0]["lr"]
                )
            )

            train_acc = unlearn_iter_func(
                data_loaders, model, criterion, optimizer, epoch, args, mask, **kwargs
            )
            scheduler.step()

            print("one epoch duration:{}".format(time.time() - start_time))
            
            if args.unlearn != 'retrain':   
                save_dir = f"{args.save_dir}/{args.unlearn}/{args.unlearn_lr}/{epoch}"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                state = {"state_dict": model.state_dict()}
                utils.save_checkpoint(state, False, save_dir, args.unlearn)
                
                device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
                
                # evaluate standard accuracy
                accuracy = {}
                for name, loader in data_loaders.items():
                    if name != "val":
                        print(f"Validating {name} loader")
                        val_acc = validate(loader, model, criterion, args)
                        accuracy[name] = val_acc
                        print(f"{name} acc: {val_acc}")
                
                # evaluate knn on office-home dataset
                if args.evaluate_knn:
                    print(f"Validating kNN on Office-Home")
                    unlearned_model = utils.load_model(f"{save_dir}/{args.unlearn}checkpoint.pth.tar", device)
                    unlearned_model.to(device)
                    office_home_knn = utils.evaluate_knn(unlearned_model, args)
                    accuracy["office_home_knn"] = float(office_home_knn*100)
                    print(f"office_home_knn: {accuracy['office_home_knn']}")
                
                # evaluate cka on office-home dataset
                if args.evaluate_cka:
                    print(f"Validating CKA between retrained model and current model on Office-Home dataset")
                    retrained_model_path = args.retrained_model_path
                    retrained_model = utils.load_model(retrained_model_path, device)
                    unleanred_model = utils.load_model(f"{save_dir}/{args.unlearn}checkpoint.pth.tar", device)
                    retrained_model.to(device)
                    unleanred_model.to(device)
                    
                    office_home_dataset_path = args.office_home_dataset_path
                    full_dataset = OfficeHomeDataset(office_home_dataset_path)
                    data_loader = DataLoader(full_dataset, batch_size=512, shuffle=False, num_workers=4)
                    
                    mode = "all"
                    
                    cka = utils.evaluate_cka(retrained_model, unleanred_model, data_loader, device, mode=mode)
                    
                    if mode == "avgpool":
                        accuracy["office_home_cka_avgpool"] = float(cka['cka']*100)
                        print(f"office_home_cka_avgpool: {accuracy['office_home_cka_avgpool']}")
                    else:
                        for layer, results in cka.items():
                            accuracy[f"office_home_cka_{layer}"] = float(results['cka']*100)
                            print(f"office_home_cka_{layer}: {accuracy[f'office_home_cka_{layer}']}")
                
                if args.use_wandb:
                    metrics = {
                        "epoch": epoch,
                        f"{args.dataset}_retain_acc": accuracy["retain"],
                        f"{args.dataset}_forget_acc": accuracy["forget"],
                    }
                    
                    if args.evaluate_knn:
                        metrics["office_home_knn"] = accuracy["office_home_knn"]
                    
                    if args.evaluate_cka:
                        for key in accuracy:
                            if key.startswith("office_home_cka"):
                                metrics[key] = accuracy[key]
                    
                    wandb.log(metrics)
                    
                print(f"saved results in {save_dir}")

    return _wrapped


def iterative_unlearn(func):
    """usage:

    @iterative_unlearn

    def func(data_loaders, model, criterion, optimizer, epoch, args)"""
    return _iterative_unlearn_impl(func)
