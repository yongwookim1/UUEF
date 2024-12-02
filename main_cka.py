import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm
import os
from PIL import Image

import arg_parser
import utils
from CKA.CKA import CudaCKA

# configuration
args = arg_parser.parse_args()
utils.setup_seed(2)
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
data_dir = '/home/dataset/OfficeHomeDataset_10072016/Real World' # set your dataset path

# model paths
pretrained_model_paths = [
    "./pretrained_model/0model_SA_best159.pth.tar", # set your model path
    "./pretrained_model/retraincheckpoint100.pth.tar",
]


class OfficeHomeDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.images = []
        self.labels = []

        self.classes = sorted(os.listdir(image_folder))
        for label, cls in enumerate(self.classes):
            cls_folder = os.path.join(image_folder, cls)
            if os.path.isdir(cls_folder):
                for img_name in os.listdir(cls_folder):
                    img_path = os.path.join(cls_folder, img_name)
                    if img_path.endswith('.jpg') or img_path.endswith('.png'):
                        self.images.append(img_path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


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


class FeatureExtractor:
    def __init__(self):
        self.features = []
        
    def hook_fn(self, module, input, output):
        self.features.append(output)
        
    def clear(self):
        self.features = []


def main():
    # data setup
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = OfficeHomeDataset(data_dir, transform=data_transforms)
    data_loader = DataLoader(full_dataset, batch_size=1024, shuffle=False, num_workers=4)

    # model setup
    model = load_model(pretrained_model_paths[1], device).to(device)
    retrained_model = load_model(pretrained_model_paths[0], device).to(device)
    model.eval()
    retrained_model.eval()

    # feature extractors
    original_extractor = FeatureExtractor()
    retrained_extractor = FeatureExtractor()
    
    linear_cka = kernel_cka = linear_check = kernel_check = 0
    cuda_cka = CudaCKA(device)

    for data in tqdm(data_loader):
        img, _ = data
        img = img.to(device)

        # register hooks
        hooks = [
            model.avgpool.register_forward_hook(original_extractor.hook_fn),
            retrained_model.avgpool.register_forward_hook(retrained_extractor.hook_fn)
        ]

        with torch.no_grad():
            model(img)
            retrained_model(img)

            f_o = original_extractor.features[0].view(original_extractor.features[0].size(0), -1)
            f_r = retrained_extractor.features[0].view(retrained_extractor.features[0].size(0), -1)

            linear_cka += cuda_cka.linear_CKA(f_o, f_r)
            linear_check += cuda_cka.linear_CKA(f_o, f_o)
            kernel_cka += cuda_cka.kernel_CKA(f_o, f_r)
            kernel_check += cuda_cka.kernel_CKA(f_r, f_r)

        # cleanup
        for hook in hooks:
            hook.remove()
        original_extractor.clear()
        retrained_extractor.clear()

    # print results
    n = len(data_loader)
    print(f"Linear CKA: {linear_cka / n:.3f}")
    print(f"Kernel CKA: {kernel_cka / n:.3f}")
    print(f"Linear CKA check: {linear_check / n:.3f}")
    print(f"Kernel CKA check: {kernel_check / n:.3f}")


if __name__ == "__main__":
    main()
