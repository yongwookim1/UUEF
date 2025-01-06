from .ResNet import *
from .ResNets import *
from .VGG import *
from .VGG_LTH import *
from .ConvNeXt import *
from .ViT import *

model_dict = {
    "resnet18": resnet18,
    "resnet50": resnet50,
    "resnet20s": resnet20s,
    "resnet44s": resnet44s,
    "resnet56s": resnet56s,
    "vgg16_bn": vgg16_bn,
    "vgg16_bn_lth": vgg16_bn_lth,
    "convnext_tiny": convnext_tiny,
    "vit_tiny": vit_tiny,
}
