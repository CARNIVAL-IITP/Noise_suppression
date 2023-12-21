import librosa
import torch
import torch.nn.functional as F
import numpy as np

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

def Loss(input, pred_y, true_y, est_phase, spec_label):

   
    in_snr = snr(input, true_y)
   
    if pred_y.shape[-1] != true_y.shape[-1]:
        pred_y = pred_y[..., :true_y.shape[-1]]
        true_y = true_y[..., :pred_y.shape[-1]]
    loss_snr = torch.mean(snr(pred_y, true_y))
    snri = loss_snr - in_snr
    
    real = spec_label[..., :257, :]
    imag = spec_label[..., 257:,:]
    spec_mag = (real**2+imag**2)**0.5
    real = real/(spec_mag + 1e-8)
    imag = imag/(spec_mag + 1e-8)
    spec_phase = torch.atan2(imag, real)

    PE = torch.zeros_like(est_phase[:,0,:,:])

    for i in range(pred_y.shape[1]):
        phase_d = torch.zeros_like(est_phase[:,0,:,:])
        for j in range(i+1, pred_y.shape[1], 1):
            
            phase_d = phase_d + (1-torch.cos(est_phase[:,i,:,:]-est_phase[:,j,:,:]-(spec_phase[:,i,:,:]-spec_phase[:,j,:,:])))
        PE = PE + (spec_mag[:,i,:,:] * phase_d)
    WPE = torch.mean(PE)
    loss = -loss_snr + 50*WPE
    
    if torch.isnan(loss):
        print(loss_snr, WPE)
        print('The losses have nan values')
        exit()
    return loss, -loss_snr, WPE, snri

