import sys
import time

import torch
import utils

from .impl import iterative_unlearn

sys.path.append(".")
from imagenet import get_x_y_from_data_dict


def l1_regularization(model):
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)


def create_layer_wise_optimizer(model, base_lr, weight_decay=5e-4):
    cka_list = [0.949, 0.976, 0.889, 0.771, 0.761, 0.847]
    # scale to 0.3-1.7 range, with higher CKA -> lower lr
    max_cka = max(cka_list)
    min_cka = min(cka_list)
    cka_normalize = [0.3 + (1 - (cka - min_cka)/(max_cka - min_cka)) * 1.4 for cka in cka_list]  # will give exact 0.3-1.7 range
    param_groups = [
        {'params': model.layer1.parameters(), 'lr': base_lr * cka_normalize[0]},
        {'params': model.layer2.parameters(), 'lr': base_lr * cka_normalize[1]},
        {'params': model.layer3.parameters(), 'lr': base_lr * cka_normalize[2]},
        {'params': model.layer4.parameters(), 'lr': base_lr * cka_normalize[3]},
        {'params': model.avgpool.parameters(), 'lr': base_lr * cka_normalize[4]},
        {'params': model.fc.parameters(), 'lr': base_lr * cka_normalize[5]}
    ]
    return torch.optim.SGD(param_groups, momentum=0.9, weight_decay=weight_decay)


@iterative_unlearn
def GA_layerwise(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    train_loader = data_loaders["forget"]
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    
    # switch to train mode
    model.train()

    optimizer = create_layer_wise_optimizer(model, args.unlearn_lr, args.weight_decay)
    start = time.time()
    if args.imagenet_arch:
        device = (
            torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
        )
        
        for i, data in enumerate(train_loader):
            image, target = get_x_y_from_data_dict(data, device)


            # compute output
            output_clean = model(image)

            loss = -criterion(output_clean, target)
            optimizer.zero_grad()
            loss.backward()

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

            optimizer.step()

            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]

            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Time {3:.2f}".format(
                        epoch, i, len(train_loader), end - start, loss=losses, top1=top1
                    )
                )
                start = time.time()
    else:
        for i, (image, target) in enumerate(train_loader):
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
                )

            image = image.to(device)
            target = target.to(device)

            # compute output
            output_clean = model(image)
            loss = -criterion(output_clean, target)

            optimizer.zero_grad()
            loss.backward()

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

            optimizer.step()

            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]

            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Time {3:.2f}".format(
                        epoch, i, len(train_loader), end - start, loss=losses, top1=top1
                    )
                )
                start = time.time()

    print("train_accuracy {top1.avg:.3f}".format(top1=top1))

    return top1.avg


@iterative_unlearn
def GA_l1(data_loaders, model, criterion, optimizer, epoch, args):
    train_loader = data_loaders["forget"]

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    for i, (image, target) in enumerate(train_loader):
        if epoch < args.warmup:
            utils.warmup_lr(
                epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
            )

        image = image.to(device)
        target = target.to(device)

        # compute output
        output_clean = model(image)
        loss = -criterion(output_clean, target) + args.alpha * l1_regularization(model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output_clean.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = utils.accuracy(output.data, target)[0]

        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if (i + 1) % args.print_freq == 0:
            end = time.time()
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                "Time {3:.2f}".format(
                    epoch, i, len(train_loader), end - start, loss=losses, top1=top1
                )
            )
            start = time.time()

    print("train_accuracy {top1.avg:.3f}".format(top1=top1))

    return top1.avg
