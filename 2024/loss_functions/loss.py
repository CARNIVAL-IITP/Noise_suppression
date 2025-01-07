import librosa
import torch
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
from torch.nn import Module




EPS = 1e-6


class BinauralLoss(Module):
    def __init__(self, win_len=400,
                 win_inc=100, fft_len=512, sr=16000,
                 ild_weight=0.1, ipd_weight=1, stoi_weight=0, 
                  snr_loss_weight=1):

        super().__init__()
        self.stft = STFT(fft_len, win_inc, win_len)
        self.istft = ISTFT(fft_len, win_inc, win_len)
        
        
    def forward(self, model_output, targets):
        target_stft_l = self.stft(targets[:, 0])
        target_stft_r = self.stft(targets[:, 1])
        

        output_stft_l = self.stft(model_output[:, 0])
        output_stft_r = self.stft(model_output[:, 1])


        loss = 0

        if self.ipd_weight > 0:
            ipd_loss = ipd_loss_rads(target_stft_l, target_stft_r,
                                     output_stft_l, output_stft_r)
            bin_ipd_loss = self.ipd_weight*ipd_loss
            
            print('\n IPD Loss = ', bin_ipd_loss)
            loss += bin_ipd_loss
        
        return loss   
        
        
        
class Loss(Module):
    def __init__(self, loss_mode="SI-SNR", win_len=400,
                 win_inc=100,
                 fft_len=512,
                 win_type="hann",
                 fix=True, sr=16000,
                 STOI_weight=1,
                 SNR_weight=0.1):
        super().__init__()
        self.loss_mode = loss_mode
        self.stft = Stft(win_len, win_inc, fft_len,
                         win_type, "complex", fix=fix)
        self.stoi_loss = NegSTOILoss(sample_rate=sr)
        self.STOI_weight = STOI_weight
        self.SNR_weight = SNR_weight

    def forward(self, model_output, targets):
        if self.loss_mode == "MSE":
            b, d, t = model_output.shape
            targets[:, 0, :] = 0
            targets[:, d // 2, :] = 0
            return F.mse_loss(model_output, targets, reduction="mean") * d

        elif self.loss_mode == "SI-SNR":
            # return -torch.mean(snr_loss(model_output, targets))
            return -(snr_loss(model_output, targets))

        elif self.loss_mode == "MAE":
            gth_spec, gth_phase = self.stft(targets)
            b, d, t = model_output.shape
            return torch.mean(torch.abs(model_output - gth_spec)) * d

        elif self.loss_mode == "STOI-SNR":
            loss_batch = self.stoi_loss(model_output, targets)
            return -(self.SNR_weight*snr_loss(model_output, targets)) + self.STOI_weight*loss_batch.mean()


def l2_norm(s1, s2):
    norm = torch.sum(s1 * s2, -1, keepdim=True)
    return norm


def snr_loss(s1, s_target, eps=EPS, reduce_mean=True):
    
    e_nosie = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10 * torch.log10((target_norm) / (noise_norm + eps) + eps)
    snr_norm = snr  # /max(snr)
    if reduce_mean:
        snr_norm = torch.mean(snr_norm)

    return snr_norm


def ild_db(s1, s2, eps=EPS):
    # s1 = _avg_signal(s1, avg_mode)
    # s2 = _avg_signal(s2, avg_mode)

    l1 = 20*torch.log10(s1 + eps)
    l2 = 20*torch.log10(s2 + eps)
    ild_value = (l1 - l2)

    return ild_value


def ild_loss_db(target_stft_l, target_stft_r,
                output_stft_l, output_stft_r, avg_mode=None):
    # amptodB = T.AmplitudeToDB(stype='amplitude')

    target_ild = ild_db(target_stft_l.abs(), target_stft_r.abs())
    output_ild = ild_db(output_stft_l.abs(), output_stft_r.abs())
    mask = speechMask(target_stft_l,target_stft_r,threshold=20)
    
    ild_loss = (target_ild - output_ild).abs()
    # breakpoint()
    masked_ild_loss = ((ild_loss * mask).sum(dim=2)).sum(dim=1)/(mask.sum(dim=2)).sum(dim=1)
   
    return masked_ild_loss.mean()

def msc_loss(target_stft_l, target_stft_r,
                output_stft_l, output_stft_r):
    
    

    # Calculate the Auto-Power Spectral Density (APSD) for left and right signals
    # Calculate the Auto-Power Spectral Density (APSD) for left and right signals
    cpsd = target_stft_l * target_stft_r.conj()
    cpsd_op = output_stft_l * output_stft_r.conj()
    
    # Calculate the Aucpsd = target_stft_l * target_stft_r.conj()to-Power Spectral Density (APSD) for left and right signals
    left_apsd = target_stft_l * target_stft_l.conj()
    right_apsd = target_stft_r * target_stft_r.conj()
    
    left_apsd_op = output_stft_l * output_stft_l.conj()
    right_apsd_op = output_stft_r * output_stft_r.conj()
    
    # Calculate the Magnitude Squared Coherence (MSC)
    msc_target = torch.abs(cpsd)**2 / ((left_apsd.abs() * right_apsd.abs())+1e-8)
    msc_output = torch.abs(cpsd_op)**2 / ((left_apsd_op.abs() * right_apsd_op.abs())+1e-8)
    
    mask = speechMask(target_stft_l,target_stft_r,threshold=20)
    
    msc_error = (msc_target - msc_output).abs()
    


    # Plot the MSC values as a function of frequency
    
    
    # breakpoint()
    # masked_msc_error = ((msc_error * mask).sum(dim=2)).sum(dim=1)/(mask.sum(dim=2)).sum(dim=1)
    
    return msc_error.mean()
    

def ipd_rad(s1, s2, eps=EPS, avg_mode=None):
    # s1 = _avg_signal(s1, avg_mode)
    # s2 = _avg_signal(s2, avg_mode)

    ipd_value = ((s1 + eps)/(s2 + eps)).angle()

    return ipd_value


def ipd_loss_rads(target_stft_l, target_stft_r,
                  output_stft_l, output_stft_r, avg_mode=None):
    # amptodB = T.AmplitudeToDB(stype='amplitude')
    target_ipd = ipd_rad(target_stft_l, target_stft_r, avg_mode=avg_mode)
    output_ipd = ipd_rad(output_stft_l, output_stft_r, avg_mode=avg_mode)

    ipd_loss = ((target_ipd - output_ipd).abs())

    mask = speechMask(target_stft_l,target_stft_r, threshold=20)
    
    masked_ipd_loss = ((ipd_loss * mask).sum(dim=2)).sum(dim=1)/(mask.sum(dim=2)).sum(dim=1)
    return masked_ipd_loss.mean()

def comp_loss_old(target_stft_l,target_stft_r,output_stft_l, output_stft_r,c=0.3):
    
    # EPS = 0+1e-10j
    target_stft_l_abs = torch.nan_to_num(target_stft_l.abs(), nan=0,posinf=0,neginf=0)
    output_stft_l_abs = torch.nan_to_num(output_stft_l.abs(), nan=0,posinf=0,neginf=0)
    target_stft_r_abs = torch.nan_to_num(target_stft_r.abs(), nan=0,posinf=0,neginf=0)
    output_stft_r_abs = torch.nan_to_num(output_stft_r.abs(), nan=0,posinf=0,neginf=0)
    
    loss_l = torch.abs(torch.pow(target_stft_l_abs,c) * torch.exp(1j*(target_stft_l.angle())) - torch.pow(output_stft_l_abs,c) * torch.exp(1j*(output_stft_l.angle())))
    loss_r = torch.abs(torch.pow(target_stft_r_abs,c) * torch.exp(1j*(target_stft_r.angle())) - torch.pow(output_stft_r_abs,c) * torch.exp(1j*(output_stft_r.angle())))
    # breakpoint()
    loss_l = torch.norm(loss_l,p='nuc')
    loss_r = torch.norm(loss_r,p='nuc')
    comp_loss_value = loss_l.mean() + loss_r.mean()
    
    
    return comp_loss_value

def comp_loss(target, output, comp_exp=0.3):
    
    EPS = 1e-6
    # target = torch.nan_to_num(target, nan=0,posinf=0,neginf=0)
    # output = torch.nan_to_num(output, nan=0,posinf=0,neginf=0)
    # target = target + EPS
    # output = output + EPS
    loss_comp = (
                    output.abs().pow(comp_exp) * output / (output.abs() + EPS) 
                    - target.abs().pow(comp_exp) * target / (target.abs() + EPS) 
                    )
    
    # loss_comp = torch.nan_to_num(loss_comp, nan=0,posinf=0,neginf=0)
    # breakpoint()
    loss_comp = torch.linalg.norm(loss_comp,ord=2,dim=(1,2))
    
    # loss_comp = loss_comp.pow(2).mean()
    
    return loss_comp.mean()

def speechMask(stft_l,stft_r, threshold=15):
    # breakpoint()
    _,_,time_bins = stft_l.shape
    thresh_l,_ = (((stft_l.abs())**2)).max(dim=2) 
    thresh_l_db = 10*torch.log10(thresh_l) - threshold
    thresh_l_db=thresh_l_db.unsqueeze(2).repeat(1,1,time_bins)
    
    thresh_r,_ = (((stft_r.abs())**2)).max(dim=2) 
    thresh_r_db = 10*torch.log10(thresh_r) - threshold
    thresh_r_db=thresh_r_db.unsqueeze(2).repeat(1,1,time_bins)
    
    
    bin_mask_l = BinaryMask(threshold=thresh_l_db)
    bin_mask_r = BinaryMask(threshold=thresh_r_db)
    
    mask_l = bin_mask_l(20*torch.log10((stft_l.abs())))
    mask_r = bin_mask_r(20*torch.log10((stft_r.abs())))
    mask = torch.bitwise_and(mask_l.int(), mask_r.int())
    
    return mask



def _avg_signal(s, avg_mode):
    if avg_mode == "freq":
        return s.mean(dim=1)
    elif avg_mode == "time":
        return s.mean(dim=2)
    elif avg_mode == None:
        return s


class BinaryMask(Module):
    def __init__(self, threshold=0.5):
        super(BinaryMask, self).__init__()
        self.threshold = threshold

    def forward(self, magnitude):
        # Compute the magnitude of the complex spectrogram
        # magnitude = torch.sqrt(spectrogram[:,:,0]**2 + spectrogram[:,:,1]**2)

        # Create a binary mask by thresholding the magnitude
        mask = (magnitude > self.threshold).float()
        # breakpoint()
        return mask


class STFT(Module):
    def __init__(self, win_len=400, win_inc=100,
                 fft_len=512):
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len

        super().__init__()

    def forward(self, x):
        stft = torch.stft(x, self.fft_len, hop_length=self.win_inc,
                          win_length=self.win_len, return_complex=True)
        return stft


class ISTFT(Module):
    def __init__(self, win_len=400, win_inc=100,
                 fft_len=512):
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len

        super().__init__()

    def forward(self, x):
        istft = torch.istft(x, self.fft_len, hop_length=self.win_inc,
                            win_length=self.win_len, return_complex=False)
        return istft

def complex_mse_loss(output, target):
    return ((output - target)**2).mean(dtype=torch.complex64)

class CLinear(nn.Module):
    def __init__(self, size_in, size_out):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(size_in, size_out, dtype=torch.complex64))
        self.bias = nn.Parameter(torch.zeros(size_out, dtype=torch.complex64))

    def forward(self, x):
        if not x.dtype == torch.complex64: x = x.type(torch.complex64)
        return x@self.weights + self.bias
    
    
    
    

import matplotlib.pyplot as plt

# def magnitude_squared_coherence(left_signal, right_signal, n_fft=1024, hop_length=256):
#     # ... (code for calculating MSC, as previously shown) ...

# # Example usage


# msc = msc_loss(left_signal, right_signal)

# # Create a frequency axis for the plot (assuming a sample rate of 44100 Hz)

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

    return -torch.mean(20 * torch.log10((vec_l2norm(t)+1e-8) / (vec_l2norm(n)+1e-8)))



def ipd(x, s):
    ipd = torch.zeros(s[0].shape)
    p_ref_s = s[0]
    p_ref_x = s[0]
    for i in range(1, 8, 1):
        ipd_s = p_ref_s-s[i]
        ipd_s = torch.cos(torch.mod(ipd_s + torch.pi, 2 * torch.pi) - torch.pi)

        ipd_x = p_ref_x-x[i]
        ipd_x = torch.cos(torch.mod(ipd_x + torch.pi, 2 * torch.pi) - torch.pi)

        d_ipd = torch.abs(ipd_s - ipd_x)
        ipd+=d_ipd
        
    ipd_avg = torch.mean(ipd / 7)
    return ipd_avg


def ild(x, s):
    # if len(x.shape)>2:
    #     b, ch, t = x.shape
    #     x = x.reshape(-1, t)
    #     s = s.reshape(-1, t)
    b, ch, t = x.shape
    ild = torch.zeros(b).to(x.device)
    m_ref_s = 20*torch.log10(torch.norm(s[:, 0, :], 2, dim=-1)+1e-8)
    m_ref_x = 20*torch.log10(torch.norm(x[:, 0, :], 2, dim=-1)+1e-8)
    
    for i in range(1, ch, 1):
        ild_s = m_ref_s - 20*torch.log10(torch.norm(s[:, i, :], 2, dim=-1)+1e-8)
        ild_x = m_ref_x - 20*torch.log10(torch.norm(x[:, i, :], 2, dim=-1)+1e-8)

        ild += torch.abs(ild_s-ild_x)

    ild_avg = torch.mean(ild / (ch-1))
    return ild_avg


def wpe(x, s, win_size=256, n_fft=512, hop_size=128, center=True):

    if len(x.shape)>2:
        b, ch, t = x.shape
        x = x.reshape(-1, t)
        s = s.reshape(-1, t)
    eps = 1e-8
    hann_window = torch.hann_window(win_size).to(x.device)


    stft_x = torch.stft(x, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                           center=center, pad_mode='reflect', return_complex=True)
    real_x = stft_x.real
    imag_x = stft_x.imag
    phase_x = torch.atan2(imag_x+eps, real_x+eps).reshape(b, ch, n_fft//2+1, -1)

    stft_s = torch.stft(s, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                           center=center, pad_mode='reflect',normalized=True, return_complex=True)
    real_s = stft_s.real
    imag_s = stft_s.imag
    phase_s = torch.atan2(imag_s+eps, real_s+eps).reshape(b, ch, n_fft//2+1, -1)


    # phase_s = torch.angle(stft_s+eps).reshape(b, ch, n_fft//2+1, -1)
    mag_s = torch.abs(stft_s).reshape(b, ch, n_fft//2+1, -1)


    PE = torch.zeros_like(phase_x[:,0,:,:])

    for i in range(ch):
        phase_d = torch.zeros_like(phase_x[:,0,:,:])
        for j in range(i+1, ch, 1):
            phase_d = phase_d + (1-torch.cos(phase_x[:,i,:,:]-phase_x[:,j,:,:]-(phase_s[:,i,:,:]-phase_s[:,j,:,:])))
        PE = PE + (mag_s[:,i,:,:] * phase_d)
    WPE = torch.mean(PE)
    return WPE

# def Loss(input, pred_y, true_y, est_phase, spec_phase):

   
#     in_snr = snr(input, true_y)
   
#     if pred_y.shape[-1] != true_y.shape[-1]:
#         pred_y = pred_y[..., :true_y.shape[-1]]
#         true_y = true_y[..., :pred_y.shape[-1]]
#     loss_snr = torch.mean(snr(pred_y, true_y))
#     snri = loss_snr - in_snr
    
#     real = spec_label[..., :257, :]
#     imag = spec_label[..., 257:,:]
#     spec_mag = (real**2+imag**2)**0.5
#     real = real/(spec_mag + 1e-8)
#     imag = imag/(spec_mag + 1e-8)
#     spec_phase = torch.atan2(imag, real)

    
#     loss = -loss_snr + 50*WPE
    
#     if torch.isnan(loss):
#         print(loss_snr, WPE)
#         print('The losses have nan values')
#         exit()
#     return loss, -loss_snr, WPE, snri

