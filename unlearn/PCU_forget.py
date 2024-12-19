import sys
import time

import torch
import utils

from .impl import iterative_unlearn

sys.path.append(".")
from imagenet import get_x_y_from_data_dict


class ProjectionHead(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers-1) + [output_dim]
        
        for i in range(len(dims)-1):
            layers.append(torch.nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(torch.nn.ReLU())
        
        self.layers = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)


@iterative_unlearn
def PCU_forget(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    forget_loader = data_loaders["forget"]
    
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # initialize proxy vectors and projectors
    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
    
    # modified proxy initialization - transpose the weights to match dimensions
    feature_dim = model.fc.weight.size(1)
    num_classes = model.fc.weight.size(0)
    
    logit_proxy = model.fc.weight.data.clone().to(device)
    logit_proxy = torch.nn.functional.normalize(logit_proxy, dim=1)
    logit_proxy = torch.nn.Parameter(logit_proxy)
    
    # modified projection heads with correct dimensions
    emb_proj = ProjectionHead(feature_dim, 512, 512, num_layers=3).to(device)
    proxy_proj = ProjectionHead(feature_dim, 512, 512, num_layers=1).to(device)
    
    # add projection heads to optimizer
    proxy_optimizer = torch.optim.Adam([logit_proxy] + 
                                     list(emb_proj.parameters()) + 
                                     list(proxy_proj.parameters()), lr=1e-3)

    model.train()
    start = time.time()

    if args.imagenet_arch:
        for i, forget_data in enumerate(forget_loader):
            forget_images, forget_targets = get_x_y_from_data_dict(forget_data, device)
            
            # capture initial features for regularization
            features = []
            def hook_fn(module, input, output):
                features.append(output)
            hooks = []
            hooks.append(model.avgpool.register_forward_hook(hook_fn))
            
            # forward pass
            forget_outputs = model(forget_images)
            
            # cross entropy loss
            ce_loss = -criterion(forget_outputs, forget_targets)
            
            # get embeddings and normalize
            forget_embeddings = features[0].view(forget_images.size(0), -1)
            forget_normalized = torch.nn.functional.normalize(forget_embeddings, dim=1)
            
            features.clear()
            for hook in hooks:
                hook.remove()
            
            # stronger temperature scaling
            temperature = 1.0
            
            # projection
            proj_forget_outputs = emb_proj(forget_embeddings)
            proj_logit_proxy = proxy_proj(logit_proxy)
            
            # projected CE loss
            proj_forget_norm = torch.nn.functional.normalize(proj_forget_outputs, dim=1)
            proj_proxy_norm = torch.nn.functional.normalize(proj_logit_proxy, dim=1)
            
            proj_similarities = torch.matmul(proj_forget_norm, proj_proxy_norm.T) / temperature
            proj_loss = -torch.nn.functional.cross_entropy(proj_similarities, forget_targets)
 
            # enhanced negative pair contrastive loss
            similarity_matrix = torch.matmul(forget_normalized, forget_normalized.T) / temperature
            
            labels_matrix = forget_targets.unsqueeze(0) == forget_targets.unsqueeze(1)
            mask = ~torch.eye(forget_targets.size(0), device=device).bool()
            
            neg_mask = labels_matrix & mask
            pos_mask = ~labels_matrix & mask
            
            exp_sim = torch.exp(similarity_matrix)
            log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
            
            contrast_loss = (-log_prob * pos_mask.float()).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)
            contrast_loss = contrast_loss.mean()

            # combine losses with stronger weighting on unlearning objectives
            total_loss = ce_loss + proj_loss + contrast_loss

            # update parameters
            optimizer.zero_grad()
            proxy_optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            proxy_optimizer.step()

            # normalize proxies
            with torch.no_grad():
                logit_proxy.data = torch.nn.functional.normalize(logit_proxy.data, dim=1)

            prec1 = utils.accuracy(forget_outputs.data, forget_targets)[0]
            losses.update(total_loss.item(), forget_images.size(0))
            top1.update(prec1.item(), forget_images.size(0))

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
    return top1.avg