import librosa

from utils import T, norm, norm_1d
from .loss_dc import loss_dc
import torch
import torch.nn.functional as F
import numpy as np
from .pmsqe_torch_8k import PMSQE

def si_snr(x, s, remove_dc=True):
    """
    Compute SI-SNR
    Arguments:
        x: vector, enhanced/separated signal
        s: vector, reference signal(ground truth)
        dimension: (batch, channel, time)
    """
    def vec_l2norm(x):
        return torch.norm(x, 2, dim=1)

    # zero mean, seems do not hurt results
    if remove_dc:
        x_zm = x - torch.mean(x)
        s_zm = s - torch.mean(s)
        t = torch.zeros(x_zm.shape[0], x_zm.shape[1])

        for i in range(x_zm.shape[0]):
            t[i,:] = torch.dot(x_zm[i,:], s_zm[i,:]) * s_zm[i,:] / torch.norm(s_zm[i,:],2) ** 2

        dev = x_zm.device
        t = t.to(dev)
        n = x_zm - t

    else:
        t = torch.dot(x, s) * s / vec_l2norm(s)**2
        n = x - t
    return 20 * torch.log10(vec_l2norm(t) / vec_l2norm(n))

def snr(x, s, remove_dc=False):
    """
    Compute SI-SNR
    Arguments:
        x: vector, enhanced/separated signal
        s: vector, reference signal(ground truth)
    """
    def vec_l2norm(x):
        return torch.norm(x, 2, dim=-1)

    # zero mean, seems do not hurt results
    if remove_dc:
        x_zm = x - torch.mean(x)
        s_zm = s - torch.mean(s)
        t = torch.dot(x_zm, s_zm) * s_zm / vec_l2norm(s_zm)**2
        n = x_zm - t
    else:
        t = s
        n = x - t

    return 20 * torch.log10((vec_l2norm(t)+1e-8) / (vec_l2norm(n)+1e-8))

def loss_speakerbeam_psa(output, label, est_phase, spec_label):

    pred_y = output
    true_y = label[0]
    loss_snr = torch.mean(snr(pred_y, true_y))

    real = spec_label[..., :257, :]
    imag = spec_label[..., 257:,:]
    spec_mag = (real**2+imag**2)**0.5
    real = real/(spec_mag + 1e-8)
    imag = imag/(spec_mag + 1e-8)
    spec_phase = torch.atan2(imag, real)
    PE = torch.zeros_like(est_phase)
    for i in range(output.shape[1]):
        phase_d = torch.zeros_like(est_phase)
        for j in range(i+1, output.shape[1], 1):
            phase_d = phase_d + (1-torch.cos(est_phase[:,i,:,:]-est_phase[:,j,:,:]-(spec_phase[:,i,:,:]-spec_phase[:,j,:,:])))
        PE = PE + (spec_mag[:,i,:,:] * phase_d)
    WPE = torch.mean(PE)

    loss = -loss_snr + 50*WPE

    if torch.isnan(loss):
        print('The losses have nan values')
        exit()
    return loss, -loss_snr, WPE
