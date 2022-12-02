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
from .feature_utils import *
import glob
import numpy as np
import torch
import os
import librosa


def dns_dataloader_test(model_name, feature_options, partition, cuda_option, cuda_device=None):
        return DataLoader(
            dns_dataset_test(model_name, feature_options, partition, cuda_option, cuda_device=cuda_device),
            batch_size=feature_options.batch_size,
            num_workers=6,
            shuffle=False,
        )


class dns_dataset_test(Dataset):
    def __init__(self, model_name, feature_options, partition, cuda_option, cuda_device=None):
        
        self.sampling_rate = feature_options.sampling_rate
        self.window_size = feature_options.window_size
        self.hop_size = feature_options.hop_size
        self.frame_length = feature_options.frame_length
        self.db_threshold = feature_options.db_threshold
        self.model_name = model_name
        self.cuda_option = cuda_option
        self.cuda_device = cuda_device
        self.file_list = []
        full_path = feature_options.data_path+partition+'/mixed/*.wav'
        # print(full_path)
        self.file_list = glob.glob(full_path)
        tar_path = feature_options.data_path + partition + '/clean/*.wav'
        self.tar_list = glob.glob(tar_path)


    def get_feature(self,fn, cn):

        sample_rate = self.sampling_rate
        audio_mix, _ = librosa.load(fn, mono=False, sr=sample_rate)
        audio_tar, _ = librosa.load(cn, mono=False, sr=sample_rate)


        stft_mix = get_stft(fn, self.sampling_rate, self.window_size, self.hop_size)

        stft_s1 = get_stft(cn, self.sampling_rate, self.window_size, self.hop_size)

        # base feature
        feature_mix = get_log_magnitude_speakerbeam(stft_mix)
        ang_mix = get_angle(stft_mix)
        feature_s1 = get_log_magnitude_speakerbeam(stft_s1)
        # feature_s2 = get_log_magnitude_speakerbeam(stft_s2)
        # one_hot_label
        mag_mix = np.abs(stft_mix)
        mag_s1 = np.abs(stft_s1)

        input, label, infdat = [audio_mix], [audio_tar], [feature_mix, feature_s1]

        return input, label, infdat


    def __getitem__(self, index):
        # print(self.file_list[0])
        file_name_mix = self.file_list[index]
        # exit()
        clean = '/clean_' + file_name_mix.split('_')[2]+'.wav'
        tar_name_mix = self.tar_list[index].split('/clean_')[0] + clean
        return self.get_feature(file_name_mix, tar_name_mix)


    def __len__(self):
        return len(self.file_list)
