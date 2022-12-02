"""
We need to define a batch size for training the deep clustering model.
Each batch has a shape (batch_size, 100/400, feature_dim)

For STFT:
8kHz fs
32 ms window length 32*8 = 256
8 ms window shift = 64

44kHz fs
128 ms window length 128*8 = 1024
32 ms window shift = 256
"""

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
# from .feature_utils import *
import glob
import numpy as np
import torch
import os
import librosa
import ms_estnoise
# import soundfile as sf
# from .audio import WaveReader

def sitec_dataloader(model_name, feature_options, partition, cuda_option, cuda_device=None):
        return DataLoader(
            sitec_dataset(model_name, feature_options, partition, cuda_option, cuda_device=cuda_device),
            batch_size=feature_options.batch_size,
            num_workers=feature_options.num_workers,
            shuffle=True,
        )


class sitec_dataset(Dataset):
    def __init__(self, model_name, feature_options, partition, cuda_option, cuda_device=None):
        
        self.sampling_rate = feature_options.sampling_rate
        self.window_size = feature_options.window_size
        self.hop_size = feature_options.hop_size
        self.frame_length = feature_options.frame_length
        self.db_threshold = feature_options.db_threshold
        self.model_name = model_name
        self.cuda_option = cuda_option
        self.cuda_device = cuda_device
        self.mixedfile_list = []
        self.cleanfile_list = []
        self.partition = partition
        mixed_path = feature_options.data_path+partition+'/mix/*.wav'
        clean_path = feature_options.data_path+partition+'/s1/*.wav'
        self.mixedfile_list = glob.glob(mixed_path)
        self.cleanfile_list = glob.glob(clean_path)
        
        self.estimator = ms_estnoise.estnoisems(256, 3700)
        self.stft = ms_estnoise.stft()



    def get_feature(self,fn, fc):

        sample_rate = self.sampling_rate
        audio_mix, sr = librosa.load(fn, mono=False, sr=sample_rate)
        audio_tar, _ = librosa.load(fc, mono=False, sr=sample_rate)
        if audio_mix.shape[-1] < self.frame_length:
            audio_mix = np.concatenate([np.zeros((4, self.frame_length - audio_mix.shape[-1])), audio_mix], -1)
            audio_tar = np.concatenate([np.zeros((4, self.frame_length - audio_tar.shape[-1])), audio_tar], -1)
        elif audio_mix.shape[-1] > self.frame_length:
            random_index = np.random.randint(audio_mix.shape[1] - self.frame_length)
            audio_mix = audio_mix[:, random_index:random_index + self.frame_length]
            audio_tar = audio_tar[:, random_index:random_index + self.frame_length]
        channel, length = audio_mix.shape
        shift = 1664
        frame = shift*2
        n_frame = (length - shift)//shift
        frames_input = []
        frames_label = []
        
        for f in range(n_frame):
            frames_input.append(np.expand_dims(audio_mix[:,f*shift:f*shift+frame], axis=0))
            # frames_label.append(np.expand_dims(audio_tar[:,f*shift:f*shift+frame], axis=0))
        frames = np.concatenate(frames_input)  # (n_frame, 4, 3328)
        spectrogram = self.stft.compute(frames[:,0,:], sr, 512, window_time=256/16000)  # (2375, 256)
        noise_est = np.sqrt(self.estimator.compute(spectrogram).T).reshape(frames.shape[0], 25, 256)
        noise_est = np.transpose(noise_est, (0, 2, 1)) # (92, 256, 25)
        return audio_mix, frames, audio_tar, noise_est



    def __getitem__(self, index):
        file_name_mix = self.mixedfile_list[index]
        
        clean = 'clean_' + file_name_mix.split('_')[2] + '.wav'
        
        file_name_clean = self.cleanfile_list[index].split('clean_')[0] + clean
        return self.get_feature(file_name_mix, file_name_clean)


    def __len__(self):
        return len(self.cleanfile_list)
