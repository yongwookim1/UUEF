import sys
import time
import copy

import torch
import utils

from .impl import iterative_unlearn

sys.path.append(".")
from imagenet import get_x_y_from_data_dict


def cov_mat_shrinkage(cov_mat, gamma1=3, gamma2=3, device="cpu"):
    I = torch.eye(cov_mat.shape[0]).to(device)
    V1 = torch.mean(torch.diagonal(cov_mat))
    off_diag = cov_mat.clone()
    off_diag.fill_diagonal_(0.0)
    mask = off_diag != 0.0
    V2 = (off_diag*mask).sum() / mask.sum()
    cov_mat_shrinked = cov_mat + gamma1*I*V1 + gamma2*(1-I)*V2
    return cov_mat_shrinked


def normalize_cov(cov_mat):
    sigma = torch.sqrt(torch.diagonal(cov_mat))  # standard deviations of the variables
    cov_mat = cov_mat/(torch.matmul(sigma.unsqueeze(1),sigma.unsqueeze(0)))
    return cov_mat


def mahalanobis_dist(samples, samples_lab, mean, S_inv):
    # check optimized version
    diff = F.normalize(tuckey_transf(samples), p=2, dim=-1)[:,None,:] - F.normalize(mean, p=2, dim=-1)
    right_term = torch.matmul(diff.permute(1,0,2), S_inv)
    mahalanobis = torch.diagonal(torch.matmul(right_term, diff.permute(1,2,0)),dim1=1,dim2=2)
    return mahalanobis


def distill(outputs_ret, outputs_original):
    soft_log_old = torch.nn.functional.log_softmax(outputs_original+10e-5, dim=1)
    soft_log_new = torch.nn.functional.log_softmax(outputs_ret+10e-5, dim=1)
    kl_div = torch.nn.functional.kl_div(soft_log_new+10e-5, soft_log_old+10e-5, reduction='batchmean', log_target=True)
    return kl_div

def tuckey_transf(vectors, delta=0.5):
    return torch.pow(vectors, delta)


def pairwise_cos_dist(x, y):
    """compute pairwise cosine distance between two tensors"""
    x_norm = torch.norm(x, dim=1).unsqueeze(1)
    y_norm = torch.norm(y, dim=1).unsqueeze(1)
    x = x / x_norm
    y = y / y_norm
    return 1 - torch.mm(x, y.transpose(0, 1))


def L2(embs_fgt,mu_distribs):
    embs_fgt = embs_fgt.unsqueeze(1)
    mu_distribs = mu_distribs.unsqueeze(0)
    dists=torch.norm((embs_fgt-mu_distribs),dim=2)
    return dists


def accuracy(model, loader, args):
    """return accuracy on a dataset given by the data loader."""
    correct = 0
    total = 0

    total_sc = torch.zeros((args.num_classes))
    correct_sc = torch.zeros((args.num_classes))

    pred_all = []
    target_all = []

    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available else "cpu")
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return correct / total


@iterative_unlearn
def SCAR(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    # store initial model state at the beginning of unlearning (epoch 0)
    if not hasattr(SCAR, 'original_model'):
        SCAR.original_model = copy.deepcopy(model)
        SCAR.original_model.eval()
        for param in SCAR.original_model.parameters():
            param.requires_grad = False
    
    original_model = SCAR.original_model
    
    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available else "cpu")
    
    # embeddings of retain set
    with torch.no_grad():
        ret_embs=[]
        labs=[]
        cnt=0
        for img_ret, lab_ret in retain_loader:
            img_ret, lab_ret = img_ret.to(device), lab_ret.to(device)
            
            features = []
            
            def hook_fn(module, input, output):
                features.append(output)
                
            hooks = []
            
            hooks.append(model.avgpool.register_forward_hook(hook_fn))
            
            with torch.no_grad():
                output = model(img_ret)
                logits_ret = original_model(img_ret)
            
            f = features[0].flatten(1) # (batch_size, feature_dim)
            
            ret_embs.append(f)
            labs.append(lab_ret)
            cnt+=1
            
            features.clear()
            for hook in hooks:
                hook.remove()
        ret_embs=torch.cat(ret_embs)
        labs=torch.cat(labs)
    

    # compute distribs from embeddings
    distribs=[]
    cov_matrix_inv =[]
    for i in range(args.num_classes):
        if type(args.class_to_replace) is list:
            if i not in args.class_to_replace:
                samples = tuckey_transf(ret_embs[labs==i])
                distribs.append(samples.mean(0))
                cov = torch.cov(samples.T)
                cov_shrinked = cov_mat_shrinkage(cov, 3, 3, device)
                cov_shrinked = normalize_cov(cov_shrinked).cpu()
                cov_matrix_inv.append(torch.linalg.pinv(cov_shrinked))

    distribs = torch.stack(distribs)
    cov_matrix_inv = torch.stack(cov_matrix_inv)
    
    # unlearn_lr = 5e-4, weight_decay = 0

    init = True
    flag_exit = False
    all_closest_class = []
    
    
    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]
    distill_loader = retain_loader
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # switch mode
    model.train()

    start = time.time()

    for n_batch, (img_fgt, lab_fgt) in enumerate(forget_loader):
        for n_batch_ret, all_batch in enumerate(retain_loader):
            img_ret, lab_ret = all_batch
            
            img_ret, lab_ret,img_fgt, lab_fgt = img_ret.to(device), lab_ret.to(device),img_fgt.to(device), lab_fgt.to(device)
            optimizer.zero_grad()

            features = []
            
            def hook(module, input, output):
                features.append(output)
                
            hooks = []
            
            hooks.append(model.avgpool.register_forward_hook(hook))
            
            with torch.no_grad():
                output = model(img_ret)
            
            embs_fgt = features[0].flatten(1)
            
            features.clear()
            
            for hook in hooks:
                hook.remove()

            # compute Mahalanobis distance between embeddings and cluster
            dists = mahalanobis_dist(embs_fgt,lab_fgt,distribs,cov_matrix_inv).T  

            if init and n_batch_ret == 0:
                closest_class = torch.argsort(dists, dim=1)
                tmp = closest_class[:, 0]
                closest_class = torch.where(tmp == lab_fgt, closest_class[:, 1], tmp)
                all_closest_class.append(closest_class)
                closest_class = all_closest_class[-1]
            else:
                closest_class = all_closest_class[n_batch]

            dists = dists[torch.arange(dists.shape[0]), closest_class[:dists.shape[0]]]

            loss_fgt = torch.mean(dists) * 1

            with torch.no_grad():
                outputs_original = original_model(img_ret)
                outputs_ret = model(img_ret)
                label_out = torch.argmax(outputs_original,dim=1)
                outputs_original = outputs_original[label_out!=args.class_to_replace[0],:]
                outputs_original[:,torch.tensor(args.class_to_replace, dtype=torch.int64)] = torch.min(outputs_original)
                
                outputs_ret = outputs_ret[label_out!=args.class_to_replace[0],:]
            
            temperature = 1
            
            loss_ret = distill(outputs_ret, outputs_original / temperature) * 5
            loss = loss_ret+loss_fgt
            
            if n_batch_ret > 900:
                del loss,loss_ret,loss_fgt, embs_fgt,dists
                break
            
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                curr_acc = accuracy(model, forget_loader)
                if curr_acc < 0.01 and epoch > 1:
                    flag_exit = True

            if flag_exit:
                break
        if flag_exit:
            break

        # evaluate accuracy on forget set every batch
        with torch.no_grad():
            model.eval()
            curr_acc = accuracy(model, forget_loader)
            if curr_acc < 0.01 and epoch > 1:
                flag_exit = True

        if flag_exit:
            break

        init = False
