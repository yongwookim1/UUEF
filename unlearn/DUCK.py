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
    """compute pairwise cosine distance between two tensors"""
    x_norm = F.normalize(x, p=2, dim=1)
    y_norm = F.normalize(y, p=2, dim=1)
    return 1 - torch.mm(x_norm, y_norm.t())


class ConvNeXtFeatureExtractor(nn.Module):
    """handle convnext's normalization and feature extraction"""
    def __init__(self, model):
        super().__init__()
        self.normalize = model.normalize
        self.forward_features = model.forward_features
        
    def forward(self, x):
        if self.normalize:
            x = self.normalize(x)
        return self.forward_features(x)


@iterative_unlearn
def DUCK(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # initialization phase for convnext models
    if not hasattr(DUCK, 'original_model'):
        DUCK.original_model = copy.deepcopy(model)
        DUCK.original_model.eval()
        for param in DUCK.original_model.parameters():
            param.requires_grad = False

        print("[DUCK] computing retain set centroids from original convnext model")
        start_centroid = time.time()
        retain_loader = data_loaders["retain"]

        # convnext-specific feature extractor setup
        if args.arch == 'convnext_tiny':
            bbone_original = ConvNeXtFeatureExtractor(DUCK.original_model)
        else:
            if args.arch != 'ViT':
                children = list(DUCK.original_model.children()[:-1])
                bbone_original = nn.Sequential(*children, nn.Flatten())

        # centroid computation logic
        class_sum = [None]*args.num_classes
        class_count = [0]*args.num_classes

        if type(args.class_to_replace) != list and not args.class_to_replace.isdigit():
            class_file = f"./class_to_replace/{args.class_to_replace}.txt"
            with open(class_file, "r") as f:
                args.class_to_replace = [int(line.strip()) for line in f if line.strip()]
        
        with torch.no_grad():
            for data in retain_loader:
                img_ret, lab_ret = get_x_y_from_data_dict(data, device)
                
                # convnext feature extraction
                if args.arch == 'convnext_tiny':
                    logits_ret = bbone_original(img_ret)
                else:
                    logits_ret = bbone_original(img_ret)

                # class exclusion handling
                for i in range(lab_ret.size(0)):
                    c = lab_ret[i].item()
                    if 0 <= c < args.num_classes and (c not in args.class_to_replace):
                        emb = logits_ret[i]
                        class_sum[c] = emb if class_sum[c] is None else class_sum[c] + emb
                        class_count[c] += 1

        DUCK.centroids = torch.stack([
            (class_sum[c]/class_count[c] if class_count[c] > 0 
             else torch.zeros_like(class_sum[0])) 
            for c in range(args.num_classes) if c not in args.class_to_replace
        ])
        print(f"[DUCK] centroid computation time: {time.time()-start_centroid:.2f}s")

    # training configuration for convnext
    model.train()
    crossent_criterion = nn.CrossEntropyLoss(label_smoothing=0.2 if args.dataset == 'tinyImagenet' else 0.0)
    
    # convnext architecture components
    if args.arch == 'convnext_tiny':
        bbone = ConvNeXtFeatureExtractor(model)
        fc = model.head
    else:
        if args.arch != 'ViT':
            children = list(model.children()[:-1])
            bbone = nn.Sequential(*children, nn.Flatten())
            fc = model.fc if hasattr(model, 'fc') else model.classifier

    # convnext-optimized hyperparameters
    args.lambda_1 = 1.2  # reduced spatial regularization
    args.lambda_2 = 1.8  # increased retain loss weight
    args.temperature = 1.5  # adjusted temperature

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # main training loop
    for i, (img_fgt, lab_fgt) in enumerate(data_loaders["forget"]):
        img_fgt, lab_fgt = img_fgt.to(device), lab_fgt.to(device)
        
        for j, (img_ret, lab_ret) in enumerate(data_loaders["retain"]):
            img_ret, lab_ret = img_ret.to(device), lab_ret.to(device)
            
            optimizer.zero_grad()
            
            # forget set processing with convnext features
            logits_fgt = bbone(img_fgt)
            dists = pairwise_cos_dist(logits_fgt, DUCK.centroids)
            closest_centroids = torch.argmin(dists, dim=1)
            loss_fgt = torch.mean(dists[torch.arange(dists.size(0)), closest_centroids]) * args.lambda_1
            
            # retain set processing
            logits_ret = bbone(img_ret)
            outputs_ret = fc(logits_ret) if fc is not None else logits_ret
            loss_ret = crossent_criterion(outputs_ret/args.temperature, lab_ret) * args.lambda_2
            
            # combined loss calculation
            loss = loss_fgt + loss_ret
            loss.backward()
            
            # gradient masking for convnext
            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None: 
                        param.grad *= mask[name]
            
            optimizer.step()
            
            # metric tracking
            prec1 = utils.accuracy(outputs_ret, lab_ret)[0]
            losses.update(loss.item(), img_ret.size(0))
            top1.update(prec1.item(), img_ret.size(0))

        # progress reporting
        if (i+1) % args.print_freq == 0:
            print(f"epoch [{epoch}][{i+1}/{len(data_loaders['forget'])}]\t"
                  f"loss {losses.avg:.4f}\taccuracy {top1.avg:.3f}")

    return top1.avg
