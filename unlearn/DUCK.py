import sys
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import utils
from .impl import iterative_unlearn

sys.path.append(".")
from imagenet import get_x_y_from_data_dict


def pairwise_cos_dist(x, y):
    """Compute pairwise cosine distance between two tensors."""
    x_norm = torch.norm(x, dim=1).unsqueeze(1)
    y_norm = torch.norm(y, dim=1).unsqueeze(1)
    x = x / x_norm
    y = y / y_norm
    return 1 - torch.mm(x, y.transpose(0, 1))


@iterative_unlearn
def DUCK(data_loaders, model, criterion, optimizer, epoch, args, mask=None):

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    if not hasattr(DUCK, 'original_model'):
        DUCK.original_model = copy.deepcopy(model)
        DUCK.original_model.eval()
        for param in DUCK.original_model.parameters():
            param.requires_grad = False

        print("[DUCK] Computing retain set centroids from the original model (epoch=0).")
        start_centroid_time = time.time()
        retain_loader = data_loaders["retain"]

        if args.arch != 'ViT':
            children = list(DUCK.original_model.children())
            bbone_original = nn.Sequential(*(children[:-1]), nn.Flatten())
        else:
            bbone_original = DUCK.original_model

        class_sum = []
        class_count = []
        for c in range(args.num_classes):
            class_sum.append(None)
            class_count.append(0)

        with torch.no_grad():
            for data in retain_loader:
                img_ret, lab_ret = data
                img_ret = img_ret.to(device)
                lab_ret = lab_ret.to(device)

                if args.arch == 'ViT':
                    logits_ret = bbone_original.forward_encoder(img_ret)
                else:
                    logits_ret = bbone_original(img_ret)

                for i in range(lab_ret.size(0)):
                    c = lab_ret[i].item()

                    if c < 0 or c >= args.num_classes:
                        continue
                    if type(args.class_to_replace) != list:
                        if not args.class_to_replace.isdigit():
                            class_file = f"./class_to_replace/{args.class_to_replace}.txt"
                            with open(class_file, "r") as f:
                                args.class_to_replace = [int(line.strip()) for line in f if line.strip()]
                    if (
                        hasattr(args, 'class_to_replace') 
                        and args.class_to_replace is not None
                        and isinstance(args.class_to_replace, list)
                        and (c in args.class_to_replace)
                    ):
                        continue

                    emb_vec = logits_ret[i]
                    
                    if class_sum[c] is None:
                        class_sum[c] = torch.zeros_like(emb_vec, device=device)

                    class_sum[c] += emb_vec
                    class_count[c] += 1

        centroids = []
        for c in range(args.num_classes):
            if (
                hasattr(args, 'class_to_replace') 
                and args.class_to_replace is not None
                and isinstance(args.class_to_replace, list)
                and (c in args.class_to_replace)
            ):
                continue

            if class_count[c] > 0 and (class_sum[c] is not None):
                avg_vec = class_sum[c] / class_count[c]
                centroids.append(avg_vec)
            else:
                if all(s is None for s in class_sum):
                    print(f"Warning: class {c} has no data and no feature_dim known.")
                    centroids.append(torch.tensor(0.0, device=device))
                else:
                    example_sum = next(s for s in class_sum if s is not None)
                    avg_vec = torch.zeros_like(example_sum)
                    centroids.append(avg_vec)

        if len(centroids) > 0:
            DUCK.centroids = torch.stack(centroids, dim=0)
        else:
            DUCK.centroids = None

        end_centroid_time = time.time()
        print(f"[DUCK] Time taken to compute retain set centroids: {end_centroid_time - start_centroid_time:.2f} seconds.")



    original_model = DUCK.original_model
    centroids = DUCK.centroids

    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]

    model.train()
    if args.dataset == 'tinyImagenet':
        ls = 0.2
    else:
        ls = 0.0
    crossent_criterion = nn.CrossEntropyLoss(label_smoothing=ls)

    if args.arch != 'ViT':
        children = list(model.children())
        bbone = nn.Sequential(*(children[:-1]), nn.Flatten())
        fc = getattr(model, 'fc', None)
        if fc is None and hasattr(model, 'classifier'):
            fc = model.classifier
    else:
        bbone = model
        fc = getattr(model, 'heads', None)

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    init = True
    all_closest_centroids = []
    flag_exit = False

    start = time.time()
    args.lambda_1 = 1.5
    args.lambda_2 = 1.5
    args.temperature = 2

    for i, (img_fgt, lab_fgt) in enumerate(forget_loader):
        img_fgt = img_fgt.to(device)
        lab_fgt = lab_fgt.to(device)

        for j, (img_ret, lab_ret) in enumerate(retain_loader):
            img_ret = img_ret.to(device)
            lab_ret = lab_ret.to(device)

            optimizer.zero_grad()

            if args.arch == 'ViT':
                logits_fgt = bbone.forward_encoder(img_fgt)
            else:
                logits_fgt = bbone(img_fgt)

            if centroids is not None:
                dists = pairwise_cos_dist(logits_fgt, centroids)
            else:
                dists = torch.zeros(logits_fgt.shape[0], device=device)

            if init:
                closest_centroids = torch.argsort(dists, dim=1)
                tmp = closest_centroids[:, 0]
                closest_centroids = torch.where(tmp == lab_fgt, closest_centroids[:, 1], tmp)
                all_closest_centroids.append(closest_centroids)
                used_closest = all_closest_centroids[-1]
            else:
                if i < len(all_closest_centroids):
                    used_closest = all_closest_centroids[i]
                else:
                    used_closest = None

            if used_closest is not None and dists.ndim == 2:
                dists_val = dists[torch.arange(dists.shape[0]), used_closest[:dists.shape[0]]]
            else:
                dists_val = torch.zeros(logits_fgt.size(0), device=device)

            loss_fgt = torch.mean(dists_val) * args.lambda_1

            if args.arch == 'ViT':
                logits_ret = bbone.forward_encoder(img_ret)
            else:
                logits_ret = bbone(img_ret)

            if fc is not None:
                outputs_ret = fc(logits_ret)
            else:
                outputs_ret = logits_ret

            loss_ret = crossent_criterion(outputs_ret / args.temperature, lab_ret) * args.lambda_2

            loss = loss_fgt + loss_ret
            loss.backward()

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

            optimizer.step()

            with torch.no_grad():
                prec1 = utils.accuracy(outputs_ret.data, lab_ret)[0]
                losses.update(loss.item(), img_ret.size(0))
                top1.update(prec1.item(), img_ret.size(0))

            if j > getattr(args, 'batch_fgt_ret_ratio', 0):
                break

        init = False

        if flag_exit:
            break

        if (i + 1) % args.print_freq == 0:
            end = time.time()
            print(
                "epoch: [{0}][{1}/{2}]\t"
                "loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                "time {3:.2f}".format(
                    epoch, i, len(forget_loader), end - start, loss=losses, top1=top1
                )
            )
            start = time.time()
    train_acc = top1.avg
    print(f"train_accuracy {train_acc:.3f}")
    return train_acc