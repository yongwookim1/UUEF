import copy
import os
from collections import OrderedDict

import arg_parser
import evaluation
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.models as models
import unlearn
import utils
from trainer import validate
from CKA.CKA import CudaCKA
from tqdm import tqdm


def evaluate_cka(original_model, unlearned_model, data_loader, device):
    original_model.eval()
    unlearned_model.eval()
    
    class FeatureExtractor:
        def __init__(self):
            self.features = []
            
        def hook_fn(self, module, input, output):
            self.features.append(output)
            
        def clear(self):
            self.features = []

    original_extractor = FeatureExtractor()
    unlearned_extractor = FeatureExtractor()
    
    linear_cka = kernel_cka = linear_check = kernel_check = 0
    cuda_cka = CudaCKA(device)

    for data, _ in tqdm(data_loader, desc="Evaluating CKA"):
        data = data.to(device)

        # register hooks
        hooks = [
            original_model.avgpool.register_forward_hook(original_extractor.hook_fn),
            unlearned_model.avgpool.register_forward_hook(unlearned_extractor.hook_fn)
        ]

        with torch.no_grad():
            original_model(data)
            unlearned_model(data)

            f_o = original_extractor.features[0].view(original_extractor.features[0].size(0), -1)
            f_u = unlearned_extractor.features[0].view(unlearned_extractor.features[0].size(0), -1)

            linear_cka += cuda_cka.linear_CKA(f_o, f_u)
            linear_check += cuda_cka.linear_CKA(f_o, f_o)
            kernel_cka += cuda_cka.kernel_CKA(f_o, f_u)
            kernel_check += cuda_cka.kernel_CKA(f_u, f_u)

        # cleanup
        for hook in hooks:
            hook.remove()
        original_extractor.clear()
        unlearned_extractor.clear()

    n = len(data_loader)
    results = {
        "linear_cka": linear_cka / n,
        "kernel_cka": kernel_cka / n,
        "linear_check": linear_check / n,
        "kernel_check": kernel_check / n
    }
    
    return results


def load_model(pretrained_model_path, device):
    print(f"\nLoading model: {pretrained_model_path}")
    model = models.resnet50(weights=None)
    checkpoint = torch.load(pretrained_model_path, map_location=device)
    if "state_dict" in checkpoint.keys():
        checkpoint = checkpoint["state_dict"]
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    state_dict = {k: v for k, v in state_dict.items() if not (k.startswith('normalize.'))}
    model.load_state_dict(state_dict, strict=True)
    print(f"Model loading complete: {pretrained_model_path}")
    return model


def main():
    args = arg_parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        utils.setup_seed(args.seed)
    seed = args.seed

    # prepare dataset
    if args.dataset == 'imagenet':
        model, retain_loader, forget_loader, val_loader = utils.setup_model_dataset(args)
    else:
        model, train_loader_full, val_loader, test_loader, marked_loader = utils.setup_model_dataset(args)
    model.to(device)

    def replace_loader_dataset(dataset, batch_size=args.batch_size, seed=1, shuffle=True):
        utils.setup_seed(seed)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
            shuffle=shuffle,
        )

    # setup forget and retain datasets
    if args.dataset == 'imagenet':
        forget_dataset = copy.deepcopy(forget_loader.dataset)
        forget_loader = replace_loader_dataset(forget_loader.dataset, seed=seed, shuffle=True)
        retain_loader = replace_loader_dataset(retain_loader.dataset, seed=seed, shuffle=True)
    else:
        forget_dataset = copy.deepcopy(marked_loader.dataset)
        try:
            marked = forget_dataset.targets < 0
            forget_dataset.data = forget_dataset.data[marked]
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
            forget_loader = replace_loader_dataset(forget_dataset, seed=seed, shuffle=True)
            
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            marked = retain_dataset.targets >= 0
            retain_dataset.data = retain_dataset.data[marked]
            retain_dataset.targets = retain_dataset.targets[marked]
            retain_loader = replace_loader_dataset(retain_dataset, seed=seed, shuffle=True)
            
            assert len(forget_dataset) + len(retain_dataset) == len(train_loader_full.dataset)
        except:
            marked = forget_dataset.targets < 0
            forget_dataset.imgs = forget_dataset.imgs[marked]
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
            forget_loader = replace_loader_dataset(forget_dataset, seed=seed, shuffle=True)
            
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            marked = retain_dataset.targets >= 0
            retain_dataset.imgs = retain_dataset.imgs[marked]
            retain_dataset.targets = retain_dataset.targets[marked]
            retain_loader = replace_loader_dataset(retain_dataset, seed=seed, shuffle=True)
            
            assert len(forget_dataset) + len(retain_dataset) == len(train_loader_full.dataset)

    print(f"Number of retain dataset: {len(retain_loader.dataset)}")
    print(f"Number of forget dataset: {len(forget_loader.dataset)}")

    if args.dataset == 'imagenet':
        unlearn_data_loaders = OrderedDict(
            retain=retain_loader, forget=forget_loader, val=val_loader
        )
    else:
        unlearn_data_loaders = OrderedDict(
            retain=retain_loader, forget=forget_loader, val=val_loader, test=test_loader
        )

    criterion = nn.CrossEntropyLoss()
    evaluation_result = None

    if args.resume:
        checkpoint = unlearn.load_unlearn_checkpoint(model, device, args)
        if checkpoint is not None:
            model, evaluation_result = checkpoint
    else:
        checkpoint = torch.load(args.model_path, map_location=device)
        if "state_dict" in checkpoint.keys():
            checkpoint = checkpoint["state_dict"]
        checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        
        if args.unlearn != "retrain":
            model.load_state_dict(checkpoint, strict=True)

        unlearn_method = unlearn.get_unlearn_method(args.unlearn)
        unlearn_method(unlearn_data_loaders, model, criterion, args)
        unlearn.save_unlearn_checkpoint(model, None, args)

    # load retrained model for CKA comparison instead of using original model
    retrained_model = load_model("./pretrained_model/retraincheckpoint100.pth.tar", device).to(device)
    retrained_model.eval()

    # evaluate CKA
    print("\nEvaluating CKA between retrained and unlearned models...")
    cka_results = evaluate_cka(retrained_model, model, val_loader, device)
    print(f"Linear CKA: {cka_results['linear_cka']:.3f}")
    print(f"Kernel CKA: {cka_results['kernel_cka']:.3f}")
    print(f"Linear CKA check: {cka_results['linear_check']:.3f}")
    print(f"Kernel CKA check: {cka_results['kernel_check']:.3f}")

    # save results
    if evaluation_result is None:
        evaluation_result = {}
    evaluation_result["accuracy"] = accuracy
    evaluation_result["cka"] = cka_results
    unlearn.save_unlearn_checkpoint(model, evaluation_result, args)


if __name__ == "__main__":
    main() 