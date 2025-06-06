import sys
import time
import copy

import torch
import torch.nn.functional as F
import utils

from .impl import iterative_unlearn

sys.path.append(".")
from imagenet import get_x_y_from_data_dict


@iterative_unlearn
def SPKD_IL(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    # store initial model state at the beginning of unlearning (epoch 0)
    if not hasattr(SPKD_IL, 'original_model'):
        SPKD_IL.original_model = copy.deepcopy(model)
        SPKD_IL.original_model.eval()
        for param in SPKD_IL.original_model.parameters():
            param.requires_grad = False
    
    original_model = SPKD_IL.original_model
    
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
        # get retain data iterator
        retain_iterator = iter(retain_loader)
        
        # combined unlearning and restore phase
        for i, forget_data in enumerate(forget_loader):
            try:
                retain_data = next(retain_iterator)
            except StopIteration:
                # if we run out of retain data, create new iterator
                retain_iterator = iter(retain_loader)
                retain_data = next(retain_iterator)

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
            
            # calculate SPKD similarity loss using retain data features
            f_s, f_t = features_s[1], features_t[0]
            b = f_s.size(0)
            f_s_flat = f_s.view(b, -1)
            f_t_flat = f_t.view(b, -1)
            
            similarity_s = torch.mm(f_s_flat, f_s_flat.t())
            similarity_t = torch.mm(f_t_flat, f_t_flat.t())
            
            similarity_s = F.normalize(similarity_s, p=2, dim=1)
            similarity_t = F.normalize(similarity_t, p=2, dim=1)
            
            similarity_loss = torch.norm(similarity_s - similarity_t, p='fro') / (b * b)
            
            # cleanup hooks
            features_s.clear()
            features_t.clear()
            for hook in hooks + hooks_t:
                hook.remove()

            # calculate combined loss
            forget_loss = -criterion(forget_output, forget_target)
            retain_loss = criterion(retain_output, retain_target)
            
            total_loss = 1 * forget_loss + 10 * retain_loss + 10 * similarity_loss

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