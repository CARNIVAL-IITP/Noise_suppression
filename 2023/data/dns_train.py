from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import glob
import numpy as np
import torch
from natsort import natsort
import os
import librosa
import soundfile as sf
import scipy.signal as ss
import ms_estnoise
import SPP

def dns_train_dataloader(model_name, feature_options, partition, cuda_option, cuda_device=None):
        return DataLoader(
            dns_dataset(model_name, feature_options, partition, cuda_option, cuda_device=cuda_device),
            batch_size=feature_options.batch_size,
            num_workers=feature_options.num_workers,
            shuffle=True,
            pin_memory=True
        )



class dns_dataset(Dataset):
    def __init__(self, model_name, feature_options, partition, cuda_option, cuda_device=None):
        
        self.sampling_rate = feature_options.sampling_rate
        self.window_size = feature_options.window_size
        self.hop_size = feature_options.hop_size
        self.frame_length = feature_options.frame_length
        self.db_threshold = feature_options.db_threshold
        self.model_name = model_name
        self.cuda_option = cuda_option
        self.cuda_device = cuda_device
        
        self.partition = partition
        mixed_path = feature_options.data_path+partition+'/mix/*.wav'
        clean_path = feature_options.data_path+partition+'/clean/*.wav'
        noise_path = feature_options.data_path+partition+'/noise/*.wav'
        self.mixedfile_list =  natsort.natsorted(glob.glob(mixed_path))
        
        self.cleanfile_list =  natsort.natsorted(glob.glob(clean_path))
        
        self.noisefile_list =  natsort.natsorted(glob.glob(noise_path))
        self.estimator = ms_estnoise.estnoisems(256, 499*8)
        self.stft = ms_estnoise.stft()


    def get_feature(self, mix, tar, info):

        random_index = np.random.randint(mix.shape[0] - self.frame_length)

        mix = mix[random_index:random_index + self.frame_length,:].transpose()  #(8, 64000)
        ch = mix.shape[0]

        tar = tar[random_index:random_index + self.frame_length,:].transpose()
        
        noi = mix-tar

        noi_spec = self.stft.compute(noi[:,:], 16000, 512, window_time=0.016, shift=0.008)

        spectrogram = self.stft.compute(mix[:,:], 16000, 512, window_time=0.016, shift=0.008)
        
        noise_est =  np.stack([np.sqrt(SPP.noisePowMat(spec, 16000)) for spec in spectrogram], 0)
        
        return mix, tar, noi_spec, noise_est, info.split('/')[-1], ch
            
        
        
    def __getitem__(self, index):
        mixture = sf.read(self.mixedfile_list[index], dtype='float32')[0] 
        speech = sf.read(self.cleanfile_list[index], dtype='float32')[0] 

        return self.get_feature(mixture, speech, self.mixedfile_list[index])


    def __len__(self):
        return len(self.cleanfile_list)
