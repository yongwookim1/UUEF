from .retrain import retrain
from .GA import GA,GA_l1
from .RL import RL
from .FT import FT,FT_l1
from .fisher import fisher,fisher_new
from .impl import load_unlearn_checkpoint, save_unlearn_checkpoint
from .Wfisher import Wfisher
from .SPKD import SPKD
from .SPKD_IL import SPKD_IL
from .SPKD_AL import SPKD_AL
from .SPKD_aug import SPKD_aug
from .SPKD_retrained import SPKD_retrained
from .AKD import AKD
from .AKD_aug import AKD_aug
from .AKD_IL import AKD_IL
from .AKD_AL import AKD_AL
from .RKD import RKD
from .RKD_IL import RKD_IL
from .GA_CKA_SPKD import GA_CKA_SPKD
from .GA_CKA import GA_CKA
from .GA_KD import GA_KD
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
from .DUCK import DUCK
from .PCU import PCU
from .PCU_forget import PCU_forget
from .PL_AKD import PL_AKD
from .RL_pro import RL_proximal
from .RL_imagenet import RL_imagenet
from .RL_SPKD import RL_SPKD
from .PL_SPKD import PL_SPKD
from .PL_RKD import PL_RKD
from .PL_AKD import PL_AKD
from .PL_KD import PL_KD
from .PL_SPKD_retrained import PL_SPKD_retrained
from .PL_SPKD_df import PL_SPKD_df
from .PL import PL
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
    elif name == "RL_SPKD":
        return RL_SPKD
    elif name == "PL_SPKD":
        return PL_SPKD
    elif name == "PL_RKD":
        return PL_RKD
    elif name == "PL_AKD":
        return PL_AKD
    elif name == "PL_KD":
        return PL_KD
    elif name == "PL_SPKD_retrained":
        return PL_SPKD_retrained
    elif name == "PL_SPKD_df":
        return PL_SPKD_df
    elif name == "PL":
        return PL
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
    elif name == "GA_KD":
        return GA_KD
    elif name == "SPKD":
        return SPKD
    elif name == "SPKD_IL":
        return SPKD_IL
    elif name == "SPKD_AL":
        return SPKD_AL
    elif name == "SPKD_aug":
        return SPKD_aug
    elif name == "SPKD_retrained":
        return SPKD_retrained
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
    elif name == "GA_CKA_SPKD":
        return GA_CKA_SPKD
    elif name == "GA_CKA":
        return GA_CKA
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
    elif name == "DUCK":
        return DUCK
    elif name == "PCU":
        return PCU
    elif name == "PCU_forget":
        return PCU_forget
    else:
        raise NotImplementedError(f"Unlearn method {name} not implemented!")
