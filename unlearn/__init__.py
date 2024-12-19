from .GA import GA,GA_l1
from .RL import RL
from .FT import FT,FT_l1
from .fisher import fisher,fisher_new
from .retrain import retrain
from .impl import load_unlearn_checkpoint, save_unlearn_checkpoint
from .Wfisher import Wfisher
from .baseline import GAwithKD
from .SPKD import SPKD
from .SPKD_IL import SPKD_IL
from .SPKD_AL import SPKD_AL
from .SPKD_aug import SPKD_aug
from .AKD import AKD
from .AKD_aug import AKD_aug
from .AKD_IL import AKD_IL
from .AKD_AL import AKD_AL
from .RKD import RKD
from .RKD_IL import RKD_IL
from .SCRUB import SCRUB
from .SCAR import SCAR
from .GA_noise import GA_noise
from .GA_pp import GA_pp
from .GA_mmd import GA_mmd
from .GA_softlabel import GA_softlabel
from .GA_layerwise import GA_layerwise
from .GA_sequential import GA_sequential
from .GA_freeze import GA_freeze
from .CU import CU 
from .PCU import PCU
from .PCU_forget import PCU_forget
from .PL_AKD import PL_AKD
from .RL_pro import RL_proximal
from .boundary_ex import boundary_expanding
from .boundary_sh import boundary_shrink


def raw(data_loaders, model, criterion, args, mask=None):
    pass


def get_unlearn_method(name):
    """method usage:

    function(data_loaders, model, criterion, args)"""
    if name == "raw":
        return raw
    elif name == "RL":
        return RL
    elif name == "RL_imagenet":
        return RL_imagenet
    elif name == "GA":
        return GA
    elif name == "FT":
        return FT
    elif name == "FT_l1":
        return FT_l1
    elif name == "fisher":
        return fisher
    elif name == "retrain":
        return retrain
    elif name == "fisher_new":
        return fisher_new
    elif name == "wfisher":
        return Wfisher
    elif name == "FT_prune":
        return FT_prune
    elif name == "FT_prune_bi":
        return FT_prune_bi
    elif name == "GA_prune":
        return GA_prune
    elif name == "GA_prune_bi":
        return GA_prune_bi
    elif name == "GA_l1":
        return GA_l1
    elif name == "boundary_expanding":
        return boundary_expanding
    elif name == "boundary_shrink":
        return boundary_shrink
    elif name == "RL_proximal":
        return RL_proximal
    elif name == "GAwithKD":
        return GAwithKD
    elif name == "GAwithKD_aug":
        return GAwithKD_aug
    elif name == "SPKD":
        return SPKD
    elif name == "SPKD_IL":
        return SPKD_IL
    elif name == "SPKD_AL":
        return SPKD_AL
    elif name == "SPKD_aug":
        return SPKD_aug
    elif name == "AKD":
        return AKD
    elif name == "AKD_aug":
        return AKD_aug
    elif name == "AKD_IL":
        return AKD_IL
    elif name == "AKD_AL":
        return AKD_AL
    elif name == "RKD":
        return RKD
    elif name == "RKD_IL":
        return RKD_IL
    elif name == "SCRUB":
        return SCRUB
    elif name == "SCAR":
        return SCAR
    elif name == "GA_noise":
        return GA_noise
    elif name == "GA_pp":
        return GA_pp
    elif name == "GA_mmd":
        return GA_mmd
    elif name == "GA_softlabel":
        return GA_softlabel
    elif name == "GA_layerwise":
        return GA_layerwise
    elif name == "GA_sequential":
        return GA_sequential
    elif name == "GA_freeze":
        return GA_freeze
    elif name == "CU":
        return CU
    elif name == "PCU":
        return PCU
    elif name == "PCU_forget":
        return PCU_forget
    elif name == "PL_AKD":
        return PL_AKD
    else:
        raise NotImplementedError(f"Unlearn method {name} not implemented!")
