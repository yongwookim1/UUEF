import sys
import time
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

from .impl import iterative_unlearn

sys.path.append(".")
from imagenet import get_x_y_from_data_dict


def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


class RKdAngle(nn.Module):
    def forward(self, student, teacher):
        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='mean')
        return loss


class RkdDistance(nn.Module):
    def forward(self, student, teacher):
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            mean_td = t_d[t_d>0].mean()
            t_d = t_d / mean_td

        d = pdist(student, squared=False)
        mean_d = d[d>0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction='mean')
        return loss


@iterative_unlearn
def PL_RKD(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    # store initial model state at the beginning of unlearning (epoch 0)
    if not hasattr(PL_RKD, 'original_model'):
        PL_RKD.original_model = copy.deepcopy(model)
        PL_RKD.original_model.eval()
        for param in PL_RKD.original_model.parameters():
            param.requires_grad = False
    
    original_model = PL_RKD.original_model
    
    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # switch mode
    model.train()
    
    if type(args.class_to_replace) != list:
        if not args.class_to_replace.isdigit():
            class_file = f"./class_to_replace/{args.class_to_replace}.txt"
            with open(class_file, "r") as f:
                class_to_replace = [int(line.strip()) for line in f if line.strip()]

    start = time.time()
    if args.imagenet_arch:
        # Unlearning phase
        print("Unlearning phase")
        for i, data in enumerate(forget_loader):
            image, target = get_x_y_from_data_dict(data, device)
            
            with torch.no_grad():
                output_o = original_model(image)
            output_o_modified = output_o.clone()
            output_o_modified[:, class_to_replace] = float('-inf')
            remain_target = output_o_modified.argmax(dim=1)
            
            output = model(image)
            loss = criterion(output, remain_target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            output = output.float()
            loss = loss.float()
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
                        epoch, i, len(forget_loader), end - start, loss=losses, top1=top1
                    )
                )
                start = time.time()

        # Restore phase with RKD
        print("Restore phase")
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
        
        dist_criterion = RkdDistance()
        angle_criterion = RKdAngle()
        
        for i, data in enumerate(retain_loader):
            image, target = get_x_y_from_data_dict(data, device)

            features_s = []
            features_t = []
            
            def hook_fn(module, input, output):
                features_s.append(output)
            
            def hook_fn_t(module, input, output):
                features_t.append(output)
            
            hooks = []
            hooks_t = []
            
            hooks.append(model.avgpool.register_forward_hook(hook_fn))
            hooks_t.append(original_model.avgpool.register_forward_hook(hook_fn_t))
                    
            output = model(image)
            with torch.no_grad():
                _ = original_model(image)
            
            f_s, f_t = features_s[0], features_t[0]
            f_s_flat = f_s.view(f_s.size(0), -1)
            f_t_flat = f_t.view(f_t.size(0), -1)
            
            f_s_norm = F.normalize(f_s_flat, p=2, dim=1)
            f_t_norm = F.normalize(f_t_flat, p=2, dim=1)
            
            distance_loss = dist_criterion(f_s_norm, f_t_norm)
            angle_loss = angle_criterion(f_s_norm, f_t_norm)
            
            features_s.clear()
            features_t.clear()
            
            for hook in hooks:
                hook.remove()
            for hook in hooks_t:
                hook.remove()
            
            ce_loss = criterion(output, target)
            loss = 10 * distance_loss + 10 * angle_loss + 10 * ce_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            output = output.float()
            loss = loss.float()
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
                        epoch, i, len(retain_loader), end - start, loss=losses, top1=top1
                    )
                )
                start = time.time()

    print("train_accuracy {top1.avg:.3f}".format(top1=top1))
    return top1.avg 