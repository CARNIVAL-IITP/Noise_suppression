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


def wsj0_2mix_inference_dataloader(model_name, feature_options, partition, cuda_option, cuda_device=None):
        return DataLoader(
            wsj0_2mix_dataset(model_name, feature_options, partition, cuda_option, cuda_device=cuda_device),
            batch_size=feature_options.batch_size,
            num_workers=6,
            shuffle=False,
        )


class wsj0_2mix_dataset(Dataset):
    def __init__(self, model_name, feature_options, partition, cuda_option, cuda_device=None):
        """
        The arguments:
            feature_options: a dictionary containing the feature params
            partition: can be "tr", "cv"
            model_name: can be "dc", "chimera", "chimera++", "phase"
            e.g.
            "feature_options": {
                "data_path": "/home/data/wsj0-2mix",
                "batch_size": 16,
                "frame_length": 400,
                "sampling_rate": 8000,
                "window_size": 256,
                "hop_size": 64,
                "db_threshold": 40
            }
        The returns:
            input: a tuple which follows the requirement of the loss
            label: a tuple which follows the requirement of the loss
            e.g.
            for dc loss:
                input: (feature_mix)
                label: (one_hot_label)
            for chimera loss:
                input: (feature_mix)
                label: (one_hot_label, mag_mix, mag_s1, mag_s2)
        """
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
        self.file_list = glob.glob(full_path)
        tar_path = feature_options.data_path + partition + '/clean/*.wav'
        self.tar_list = glob.glob(tar_path)
        # self.file_list = random.shuffle(glob.glob(full_path))
        # random.seed()
        # random.shuffle(self.file_list)
        # self.file_list = glob.glob(full_path)
        # print(len(file_list))
        # exit()


    def get_feature(self,fn, cn):
        sample_rate = self.sampling_rate
        audio_mix, _ = librosa.load(fn, mono=False, sr=sample_rate)
        audio_tar, _ = librosa.load(cn, mono=False, sr=sample_rate)

        input, label, infdat = [audio_mix], [audio_tar], [fn, cn]

        return input, label, infdat

    def __getitem__(self, index):
        # print(self.file_list[0])
        file_name_mix = self.file_list[index]
        clean = '/clean_' + file_name_mix.split('_')[2] + '.wav'
        tar_name_mix = self.tar_list[index].split('/clean_')[0] + clean
        return self.get_feature(file_name_mix, tar_name_mix)


    def __len__(self):
        return len(self.file_list)
