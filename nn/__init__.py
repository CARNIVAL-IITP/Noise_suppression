"""
For every model created under nn directory. Please import it here.
The model should takes a list of tensors named "input" as the input,
and also returns a list of tensors as the output.
"""

from .chimera import chimera
from .deep_clustering import deep_clustering
from .enhancement import enhance
from .phase_network import phase_net
from .speakerbeam import speakerbeam
from .dc_crn import DCCRN_mimo
from .TRUnet import TRUNet