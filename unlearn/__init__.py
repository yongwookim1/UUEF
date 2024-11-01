from .GA import GA,GA_l1
from .RL import RL
from .FT import FT,FT_l1
from .fisher import fisher,fisher_new
from .retrain import retrain
from .impl import load_unlearn_checkpoint, save_unlearn_checkpoint
from .Wfisher import Wfisher
from .baseline import GAwithKD
from .SPKD import SPKD
from .SPKD_similarity import SPKD_similarity
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
    elif name == "SPKD":
        return SPKD
    elif name == "SPKD_similarity":
        return SPKD_similarity
    else:
        raise NotImplementedError(f"Unlearn method {name} not implemented!")
