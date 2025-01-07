import numpy as np
import sys
import os


def noisePowMat(noisy, fs):
    # print(noisy.shape)  #(256, 1596)
    
    #some constants
    frLen   = int(32e-3*fs)  #frame size
    fShift  = frLen/2   # fShift size
    # nFrames = (len(noisy)/fShift)-1 #number of frames: 257
    nFrames = noisy.shape[-1] #number of frames: 1596
    
    anWin  = np.hanning(frLen) #analysis window
    
    # allocate some memory
    
    noisePowMat = np.zeros((frLen//2,nFrames))
    
    # initialize
    # noisePow = init_noise_tracker_ideal_vad(noisy,frLen,frLen,fShift, anWin) # This function computes the initial noise PSD estimate. It is assumed that the first 5 time-frames are noise-only.
   
    noisePow=np.mean(abs(noisy[:,:5])*abs(noisy[:,:5]), axis=1)
    
    noisePowMat[:, 0]=noisePow


    PH1mean  = 0.5
    alphaPH1mean = 0.9
    alphaPSD = 0.8


    #constants for a posteriori SPP
    q          = 0.5 # a priori probability of speech presence:
    priorFact  = q/(1-q)
    xiOptDb    = 15 # optimal fixed a priori SNR for SPP estimation
    xiOpt      = np.power(10,xiOptDb/10)
    logGLRFact = np.log(1/(1+xiOpt))
    GLRexp     = xiOpt/(1+xiOpt)

    for indFr in range(1, nFrames):
        
       
        noisyDftFrame =noisy[:, indFr]
       
        
        noisyPer = noisyDftFrame*np.conj(noisyDftFrame)
        
        snrPost1 =  noisyPer/(noisePow +1e-8)# a posteriori SNR based on old noise power estimate


        # noise power estimation
        
        
        # GLR     = priorFact* np.exp(min(logGLRFact + GLRexp*snrPost1,200))
        tmp = logGLRFact + GLRexp*snrPost1
        tmp[tmp>200]=200
        GLR = priorFact*np.exp(tmp)

        PH1     = GLR/(1+GLR) # a posteriori speech presence probability

        PH1mean  = alphaPH1mean * PH1mean + (1-alphaPH1mean) * PH1
        stuckInd = PH1mean > 0.99
        
        PH1[stuckInd>0.99] = 0.99
        estimate =  PH1* noisePow + (1-PH1)* noisyPer 
        noisePow = alphaPSD *noisePow+(1-alphaPSD)*estimate
        
        noisePowMat[:,indFr] = noisePow
        
    return noisePowMat




def init_noise_tracker_ideal_vad(noisy, fr_size, fft_size, hop, sq_hann_window):
    noisy_dft_frame_matrix= np.zeros(1, 5)
    for I in range(1,5):
        noisy_frame=sq_hann_window*noisy[(I-1)*hop+1:(I-1)*hop+fr_size]
        noisy_dft_frame_matrix[:,I]=np.fft(noisy_frame,fft_size)
    
    noise_psd_init=np.mean(abs(noisy_dft_frame_matrix[1:fr_size/2+1,1:])^2,2)#compute the initialisation of the noise tracking algorithms.
    return noise_psd_init

