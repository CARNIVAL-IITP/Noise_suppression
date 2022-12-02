"""
For every new loss function added, please import it here and add it to loss_fns
The loss function should take two arguments:
    output: a tuple from the network
    label: a tuple which is from dataloader
You need to assert the format of the output and label in the loss function!
"""
from .loss import loss_wpe

from attrdict import AttrDict

def get_lossfns():
    loss_fns = AttrDict()
    loss_fns["wpe"] = loss_wpe
    return loss_fns
