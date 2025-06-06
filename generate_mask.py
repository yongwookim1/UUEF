import copy
import os
from collections import OrderedDict
from tqdm.auto import tqdm

import arg_parser
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import unlearn
import utils
from imagenet import get_x_y_from_data_dict


def replace_loader_dataset(dataset, batch_size, seed=1, shuffle=True):
    utils.setup_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        shuffle=shuffle,
    )


def save_gradient_ratio(data_loaders, model, criterion, args, mode="GA"):
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.unlearn_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    gradients = {}
    forget_loader = data_loaders["forget"]
    model.eval()

    device = (f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    for name, param in model.named_parameters():
        gradients[name] = 0

    for i, data in enumerate(tqdm(forget_loader)):
        image, target = get_x_y_from_data_dict(data, device=args.gpu)
        random_target = torch.randint(0, args.num_classes, target.shape).to(device)

        # compute output
        if mode == "GA":
            output_clean = model(image)
            loss = -criterion(output_clean, target)

        elif mode == "RL":
            output_clean = model(image)
            loss = criterion(output_clean, random_target)

        optimizer.zero_grad()
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients[name] += param.grad.data

    with torch.no_grad():
        for name in gradients:
            gradients[name] = torch.abs_(gradients[name])

    threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for i in threshold_list:
        sorted_dict_positions = {}
        hard_dict = {}

        # concatenate all tensors into a single tensor
        all_elements = -torch.cat([tensor.flatten() for tensor in gradients.values()])

        # calculate the threshold index for the top elements
        threshold_index = int(len(all_elements) * i)

        # calculate positions of all elements
        positions = torch.argsort(all_elements)
        ranks = torch.argsort(positions)

        start_index = 0
        for key, tensor in gradients.items():
            num_elements = tensor.numel()
            tensor_ranks = ranks[start_index: start_index + num_elements]

            sorted_positions = tensor_ranks.reshape(tensor.shape)
            sorted_dict_positions[key] = sorted_positions

            # set the corresponding elements to 1
            threshold_tensor = torch.zeros_like(tensor_ranks)
            threshold_tensor[tensor_ranks < threshold_index] = 1
            threshold_tensor = threshold_tensor.reshape(tensor.shape)
            hard_dict[key] = threshold_tensor
            start_index += num_elements

            torch.save(hard_dict, os.path.join(args.save_dir, f"with_{i}.pt"))


def main():
    args = arg_parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        utils.setup_seed(args.seed)
    seed = args.seed

    # prepare dataset
    if args.dataset == 'imagenet':
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

    model.cuda()

    if args.dataset != "imagenet":
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
            forget_loader = replace_loader_dataset(forget_dataset, args.batch_size, seed=seed, shuffle=True)
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
            retain_loader = replace_loader_dataset(retain_dataset, args.batch_size, seed=seed, shuffle=True)
            assert len(forget_dataset) + len(retain_dataset) == len(train_loader_full.dataset)
        else:
            try:
                marked = forget_dataset.targets < 0
                forget_dataset.data = forget_dataset.data[marked]
                forget_dataset.targets = -forget_dataset.targets[marked] - 1
                forget_loader = replace_loader_dataset(forget_dataset, args.batch_size, seed=seed, shuffle=True)
                retain_dataset = copy.deepcopy(marked_loader.dataset)
                marked = retain_dataset.targets >= 0
                retain_dataset.data = retain_dataset.data[marked]
                retain_dataset.targets = retain_dataset.targets[marked]
                retain_loader = replace_loader_dataset(retain_dataset, args.batch_size, seed=seed, shuffle=True)
                assert len(forget_dataset) + len(retain_dataset) == len(train_loader_full.dataset)
            except:
                marked = forget_dataset.targets < 0
                forget_dataset.imgs = forget_dataset.imgs[marked]
                forget_dataset.targets = -forget_dataset.targets[marked] - 1
                forget_loader = replace_loader_dataset(forget_dataset, args.batch_size, seed=seed, shuffle=True)
                retain_dataset = copy.deepcopy(marked_loader.dataset)
                marked = retain_dataset.targets >= 0
                retain_dataset.imgs = retain_dataset.imgs[marked]
                retain_dataset.targets = retain_dataset.targets[marked]
                retain_loader = replace_loader_dataset(retain_dataset, args.batch_size, seed=seed, shuffle=True)
                assert len(forget_dataset) + len(retain_dataset) == len(train_loader_full.dataset)

    print(f"number of retain dataset {len(retain_loader.dataset)}")
    print(f"number of forget dataset {len(forget_loader.dataset)}")

    if args.dataset == 'imagenet':
        unlearn_data_loaders = OrderedDict(
            retain=retain_loader, forget=forget_loader
        )
    else:
        unlearn_data_loaders = OrderedDict(
            retain=retain_loader, forget=forget_loader, val=val_loader, test=test_loader
        )

    criterion = nn.CrossEntropyLoss()

    if args.resume:
        checkpoint = unlearn.load_unlearn_checkpoint(model, device, args)

    if args.resume and checkpoint is not None:
        model, evaluation_result = checkpoint
    else:
        checkpoint = torch.load(args.model_path, map_location=device)
        if "state_dict" in checkpoint.keys():
            checkpoint = checkpoint["state_dict"]
            
        checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
            
        model.load_state_dict(checkpoint, strict=True)

    save_gradient_ratio(unlearn_data_loaders, model, criterion, args, mode="GA")


if __name__ == "__main__":
    main()
