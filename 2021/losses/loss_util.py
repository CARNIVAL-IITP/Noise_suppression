from .loss_dccrn import loss_speakerbeam_psa, permute_SI_SNR
from attrdict import AttrDict



def get_lossfns():
    loss_fns = AttrDict()
    loss_fns["SNR"] = loss_SNR
    loss_fns["permute_SI_SNR"] = permute_SI_SNR
    return loss_fns
