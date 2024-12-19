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
        # N x C
        # N x N x C

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
def RKD_IL(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    # store initial model state at the beginning of unlearning (epoch 0)
    if not hasattr(RKD_IL, 'original_model'):
        RKD_IL.original_model = copy.deepcopy(model)
        RKD_IL.original_model.eval()
        for param in RKD_IL.original_model.parameters():
            param.requires_grad = False
    
    original_model = RKD_IL.original_model
    
    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]
    distill_loader = retain_loader
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # switch mode
    model.train()

    start = time.time()
    if args.imagenet_arch:
        # combined unlearning and restore phase
        for i, (forget_data, retain_data) in enumerate(zip(forget_loader, distill_loader)):
            # process forget data
            forget_image, forget_target = get_x_y_from_data_dict(forget_data, device)

            # process retain data
            retain_image, retain_target = get_x_y_from_data_dict(retain_data, device)
            
            # setup feature hooks
            features_s = []
            features_t = []
            
            def hook_fn(module, input, output):
                features_s.append(output)
                
            def hook_fn_t(module, input, output):
                features_t.append(output)
            
            hooks = [model.avgpool.register_forward_hook(hook_fn)]
            hooks_t = [original_model.avgpool.register_forward_hook(hook_fn_t)]
            
            # forward passes
            forget_output = model(forget_image)
            retain_output = model(retain_image)
            with torch.no_grad():
                _ = original_model(retain_image)
            
            # calculate RKD losses using only retain data features
            f_s, f_t = features_s[1], features_t[0]
            f_s_flat = f_s.view(f_s.size(0), -1)
            f_t_flat = f_t.view(f_t.size(0), -1)
            
            # normalize feature vectors
            f_s_norm = F.normalize(f_s_flat, p=2, dim=1)
            f_t_norm = F.normalize(f_t_flat, p=2, dim=1)
            
            # compute RKD losses using normalized features
            dist_criterion = RkdDistance()
            angle_criterion = RKdAngle()
            distance_loss = dist_criterion(f_s_norm, f_t_norm)
            angle_loss = angle_criterion(f_s_norm, f_t_norm)
            
            # cleanup hooks
            features_s.clear()
            features_t.clear()
            for hook in hooks + hooks_t:
                hook.remove()

            # calculate combined loss
            forget_loss = -criterion(forget_output, forget_target)
            retain_loss = criterion(retain_output, retain_target)
            total_loss = 1 * forget_loss + 20 * retain_loss + 20 * distance_loss + 20 * angle_loss

            # update model
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # update metrics using both retain and forget data
            retain_output = retain_output.float()
            forget_output = forget_output.float()
            loss = total_loss.float()
            
            retain_prec1 = utils.accuracy(retain_output.data, retain_target)[0]
            forget_prec1 = utils.accuracy(forget_output.data, forget_target)[0]

            # average the metrics across both datasets
            combined_prec1 = (retain_prec1 + forget_prec1) / 2
            combined_size = retain_image.size(0) + forget_image.size(0)

            losses.update(loss.item(), combined_size)
            top1.update(combined_prec1.item(), combined_size)

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

    print("train_accuracy {top1.avg:.3f}".format(top1=top1))

    return top1.avg