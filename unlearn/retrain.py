from trainer import train

from .impl import iterative_unlearn
from trainer import *
from utils import *

@iterative_unlearn
def retrain(data_loaders, model, criterion, optimizer, epoch, args, mask):
    retain_loader = data_loaders["retain"]
    return train(retain_loader, model, criterion, optimizer, epoch, args, mask)
