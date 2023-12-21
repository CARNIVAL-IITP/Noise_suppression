import sys
import numpy as np
from scipy import fftpack, signal, stats





class M:
    def __init__(self, d):
        self.d = np.array([1, 2, 5, 8, 10, 15, 20, 30, 40, 60, 80, 120, 140, 160])
        self.m = np.array([0, 0.26, 0.48, 0.58, 0.61, 0.668, 0.705,
               0.762, 0.8, 0.841, 0.865, 0.89, 0.9, 0.91])
        if np.where(self.d == d)[0]:
            self.get_m = float(self.m[np.where(self.d == d)[0]])
        else:
            intra_index_1 = intra_index_2 = None
            for i in range(len(self.d)):
                if d < self.d[i]:
                    intra_index_1 = i-1
                    intra_index_2 = i
                    break
            if not (intra_index_1 and intra_index_2):
                sys.stderr.write("WARNING: parameter D is out of range!")
                sys.exit(1)
            self._intrapolate(d, intra_index_1, intra_index_2)

    def _intrapolate(self, d, ind_1, ind_2):
        self.get_m = ((self.m[ind_2] - self.m[ind_1])*(d - self.d[ind_1])
                      )/(self.d[ind_2] - self.d[ind_1]) + self.m[ind_1]

    def get_m(self):
        return self.get_m



_D = 96
_U = 8
_V = _D//_U
_Md = M(_D)
_Mv = M(_V)

class stft:
    def __init__(self) -> None:
        self.shift = None

    def compute(self, sig, samp_rate, nfft, window_time=0.025, shift=0.00625):
        """A short time fourier transform of a time signal
        @signal -- time signal
        @samp_rate -- sample rate of the signal
        @nff -- FFT length
        @window_time -- window time
        @overlap -- overlap"""
        
        if window_time > sig.shape[-1]/samp_rate:
            sys.stderr.write(
                "Window time should be less than the time length of signal")
            sys.exit(1)
        window_length = int(samp_rate*window_time)
        self.shift = int(samp_rate*shift)

        window = signal.hamming(window_length)
        # spec = [np.expand_dims(np.abs(fftpack.fft(window*sig[:, i:i+window_length], nfft)[:, :nfft//2]),0) for i in range(0, sig.shape[-1]-(window_length-self.shift), self.shift)]
        # spec = np.transpose(np.concatenate(spec,0), (1,0,2))
        # batch, frame, freq = spec.shape
        # spec = spec.reshape(batch*frame, freq)
        
        out = np.stack([np.abs(fftpack.fft(window*sig[:, i:i+window_length], nfft)[:, :nfft//2]) for i in range(0, sig.shape[-1]-(window_length-self.shift), self.shift)],0)
        
        # print(out.transpose(1,2,0).shape)
        # exit()
        # np.concatenate([np.abs(fftpack.fft(window*sig[:, i:i+window_length], nfft)[:, :nfft//2]) for i in range(0, sig.shape[-1]-(window_length-self.shift), self.shift)],0)
        return out.transpose(1,2,0)

    def get_shift(self):
        return self.shift


def gen_wgnoise(sig):
    # snr of 30 dB
    snr_lin = 10**(30/10)
    sig_pow = sum([s**2 for s in sig])/len(sig)
    amp = np.sqrt(sig_pow/snr_lin)
    return stats.norm(0, amp)



class estnoisems:
    def __init__(self, nfft, niteration) -> None:
        self.noise_est = np.ones(nfft)*np.inf
        self.alpha_vec = np.ones(nfft)*0.96
        self.alpbuff = np.ones((niteration, nfft))*0.96
        self.psd_vec = np.ones(nfft)
        self.psdbuff = np.ones((niteration, nfft))
        self.actmin_vec = np.ones(nfft)*np.inf
        self.actmin_sub_vec = np.ones(nfft)*np.inf
        self.actbuff = np.ones((_U, nfft))*np.inf
        self.subwc = _V   # sub window

    def compute(self, Y_vec):
        global P_min_u
        """Estimation of the noise based on minimum statistics"""
        
        (niteration, nfft) = Y_vec.shape
        self.psd_vec = Y_vec[0, :]**2
        self.noise_est = self.psd_vec
        alpha_corr = 1
        fmoment = self.psd_vec
        smoment = self.psd_vec**2
        lmin_flag = np.zeros(nfft)
        ibuf = 0
        x = np.zeros((niteration, nfft))
        for n in range(niteration):
            '''compute smoothing factor'''
            Y_vec_n = Y_vec[n, :]  # consider only a frame
            # compute tilda correction factor
            talpha_corr = 1/(1 + (sum(self.psd_vec)/sum(Y_vec_n**2 + 1e-8) - 1)**2)

            tmp = np.array([talpha_corr])
            a = 0.9
            tmp[tmp < a] = a
            alpha_corr = a * alpha_corr + (1-a) * tmp

            # compute the smoothing factor
            alpha_min = np.divide(Y_vec_n**2, self.noise_est+1e-8)**(100/(0.064*16000))
            alpha_min[alpha_min > 0.3] = 0.3
            smoothed_snr = np.divide(self.psd_vec, self.noise_est + 1e-8)
            self.alpha_vec = np.divide(0.96 * alpha_corr , 1 + (smoothed_snr - 1)**2)
            self.alpha_vec[self.alpha_vec < alpha_min] = alpha_min[self.alpha_vec < alpha_min]
            # self.alpha_vec = 1 / (1 + (self.psd_vec/self.noise_est - 1)**2)

            self.alpbuff[n, :] = self.alpha_vec     # save all alpha values
            '''Smoothing'''
            # self.alpha_vec = np.ones(nfft)*0.6
            
            self.psd_vec = self.alpha_vec * self.psd_vec + (1-self.alpha_vec) * Y_vec_n**2  # compute the smoothed psd
            self.psdbuff[n, :] = self.psd_vec
            # compute beta and P_var_vec
            
            beta_vec = min_complex(self.alpha_vec ** 2, np.array([0.8]))
            fmoment = beta_vec * fmoment + (1 - beta_vec) * self.psd_vec
            smoment = beta_vec * smoment + (1 - beta_vec) * self.psd_vec ** 2
            var = smoment - (fmoment ** 2) + 1e-8

            # print(n, var)
            # compute the DOF
            DOF_vec = max_complex(
                (2 * self.noise_est ** 2) / var, np.array([2.0]))
            # compute the tDOF for windows
            tDOF_vec = (DOF_vec - 2 * _Md.get_m) / (1 - _Md.get_m)
            tDOF_sub_vec = (DOF_vec - 2 * _Mv.get_m) / (1 - _Mv.get_m)
            # compute the bias
            bias_vec = 1 + ((_D - 1) * 2 / tDOF_vec)
            bias_sub_vec = 1 + ((_V - 1) * 2 / tDOF_sub_vec)
            # compute q_inv_mean
            Q_inv_mean = sum(1/DOF_vec)/len(Y_vec)
            # compute b_corr
            B_corr = 1 + (2.12*np.sqrt(Q_inv_mean))
            k_mod = self.psd_vec*bias_vec*B_corr < self.actmin_vec

            """minimum tracking"""
            if any(k_mod):  # if present PSD's value < minimum value
                self.actmin_vec[k_mod] = self.psd_vec[k_mod] * bias_vec[k_mod]*B_corr
                self.actmin_sub_vec[k_mod] = self.psd_vec[k_mod] * bias_sub_vec[k_mod]*B_corr

            if self.subwc > 0 and self.subwc < _V:  # 현재 sub window의 처음이나 끝이 아닐 때
                lmin_flag = np.logical_or(k_mod, lmin_flag)
                P_min_u = min_complex(self.actmin_sub_vec, P_min_u)
                self.noise_est = P_min_u.copy()
                self.subwc += 1
            else:
                if self.subwc >= _V:
                    lmin_flag = np.logical_and(k_mod, lmin_flag)
                    # uses buffer for storage of the past u frames
                    ibuf = 1+(ibuf % _U)    # increment pointer
                    self.actbuff[ibuf-1, :] = self.actmin_vec.copy()
                    P_min_u = min_complex_mat(self.actbuff)
                    if Q_inv_mean < 0.03:
                        noise_slope_max = 10**(8/10)
                    elif Q_inv_mean < 0.05:
                        noise_slope_max = 10**(4/10)
                    elif Q_inv_mean < 0.06:
                        noise_slope_max = 10**(2/10)
                    else:
                        noise_slope_max = 10**(1.2/10)

                    # noise_slope_max = 100
                    # if any(np.logical_and(lmin_flag, self.actmin_sub_vec < (noise_slope_max*P_min_u))):
                    #     if any(np.logical_and(np.logical_and(lmin_flag, self.actmin_sub_vec < (noise_slope_max*P_min_u)), self.actmin_sub_vec > P_min_u)):


                    lmin =np.logical_and(np.logical_and(np.logical_and(lmin_flag, np.logical_not(k_mod)),
                                                         self.actmin_sub_vec < (noise_slope_max*P_min_u)),
                                                        self.actmin_sub_vec > P_min_u)
                    if any(np.logical_and(np.logical_and(lmin_flag, self.actmin_sub_vec < (noise_slope_max*P_min_u)), self.actmin_sub_vec > P_min_u)):
                        P_min_u[lmin] = self.actmin_sub_vec[lmin]
                        # replace all previously stored actmin with actmin_sub
                        self.actbuff[:, lmin] = np.ones((_U, 1))*P_min_u[lmin]
                    lmin_flag[:] = 0
                    self.actmin_vec[:] = np.Inf
                    self.actmin_sub_vec[:] = np.Inf
                    self.subwc = 1
            x[n, :] = self.noise_est
        return x

    def get_alpha(self):
        return self.alpbuff

    def get_smoothed(self):
        return self.psdbuff




def max_complex(a, b):
    """
    This is python implementation of [1],[2], and [3]. 

    Refs:
       [1] Rainer Martin.
           Noise power spectral density estimation based on optimal smoothing and minimum statistics.
           IEEE Trans. Speech and Audio Processing, 9(5):504-512, July 2001.
       [2] Rainer Martin.
           Bias compensation methods for minimum statistics noise power spectral density estimation
           Signal Processing, 2006, 86, 1215-1229
       [3] Dirk Mauler and Rainer Martin
           Noise power spectral density estimation on highly correlated data
           Proc IWAENC, 2006

         Copyright (C) Mike Brookes 2008
         Version: $Id: estnoisem.m 1718 2012-03-31 16:40:41Z dmb $

      VOICEBOX is a MATLAB toolbox for speech processing.
      Home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
    """
    if len(a) == 1 and len(b) > 1:
        a = np.tile(a, np.shape(b))
    if len(b) == 1 and len(a) > 1:
        b = np.tile(b, np.shape(a))

    i = np.logical_or(np.iscomplex(a), np.iscomplex(b))

    aa = a.copy()
    bb = b.copy()

    if any(i):
        aa[i] = np.absolute(aa[i])
        bb[i] = np.absolute(bb[i])
    if a.dtype == 'complex' or b.dtype == 'complex':
        cc = np.array(np.zeros(np.shape(a)))
    else:
        cc = np.array(np.zeros(np.shape(a)), dtype=float)

    i = aa > bb
    cc[i] = a[i]
    cc[np.logical_not(i)] = b[np.logical_not(i)]

    return cc


def min_complex(a, b):
    """
    This is python implementation of [1],[2], and [3]. 

    Refs:
       [1] Rainer Martin.
           Noise power spectral density estimation based on optimal smoothing and minimum statistics.
           IEEE Trans. Speech and Audio Processing, 9(5):504-512, July 2001.
       [2] Rainer Martin.
           Bias compensation methods for minimum statistics noise power spectral density estimation
           Signal Processing, 2006, 86, 1215-1229
       [3] Dirk Mauler and Rainer Martin
           Noise power spectral density estimation on highly correlated data
           Proc IWAENC, 2006

         Copyright (C) Mike Brookes 2008
         Version: $Id: estnoisem.m 1718 2012-03-31 16:40:41Z dmb $

      VOICEBOX is a MATLAB toolbox for speech processing.
      Home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
    """
    if len(a) == 1 and len(b) > 1:
        a = np.tile(a, np.shape(b))
    if len(b) == 1 and len(a) > 1:
        b = np.tile(b, np.shape(a))

    i = np.logical_or(np.iscomplex(a), np.iscomplex(b))

    aa = a.copy()
    bb = b.copy()

    if any(i):
        aa[i] = np.absolute(aa[i])
        bb[i] = np.absolute(bb[i])

    if a.dtype == 'complex' or b.dtype == 'complex':
        cc = np.zeros(np.shape(a))
    else:
        cc = np.zeros(np.shape(a), dtype=float)

    i = aa < bb     # find the indexes with minimum values below the threshold
    cc[i] = a[i]
    cc[np.logical_not(i)] = b[np.logical_not(i)]

    return cc


def min_complex_mat(a):
    """
    This is python implementation of [1],[2], and [3]. 

    Refs:
       [1] Rainer Martin.
           Noise power spectral density estimation based on optimal smoothing and minimum statistics.
           IEEE Trans. Speech and Audio Processing, 9(5):504-512, July 2001.
       [2] Rainer Martin.
           Bias compensation methods for minimum statistics noise power spectral density estimation
           Signal Processing, 2006, 86, 1215-1229
       [3] Dirk Mauler and Rainer Martin
           Noise power spectral density estimation on highly correlated data
           Proc IWAENC, 2006

         Copyright (C) Mike Brookes 2008
         Version: $Id: estnoisem.m 1718 2012-03-31 16:40:41Z dmb $

      VOICEBOX is a MATLAB toolbox for speech processing.
      Home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
    """
    s = np.shape(a)
    m = np.array(np.zeros(s[1]))
    for i in range(0, s[1]):
        j = np.argmin(np.absolute(a[:, i]))
        m[i] = a[j, i]
    return m
