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

def sitec_inference_dataloader(model_name, feature_options, partition, cuda_option, cuda_device=None):
        return DataLoader(
            sitec_dataset(model_name, feature_options, partition, cuda_option, cuda_device=cuda_device),
            batch_size=feature_options.batch_size,
            num_workers=6,
            shuffle=False,
        )


class sitec_dataset(Dataset):
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
        full_path = feature_options.data_path+partition+'/mix/*.wav'
        self.file_list = glob.glob(full_path)
        tar_path = feature_options.data_path + partition + '/s1/*.wav'
        self.tar_list = glob.glob(tar_path)
        self.estimator = ms_estnoise.estnoisems(256, 2375)
        self.stft = ms_estnoise.stft()
        # self.file_list = random.shuffle(glob.glob(full_path))
        # random.seed()
        # random.shuffle(self.file_list)
        # self.file_list = glob.glob(full_path)
        # print(len(file_list))
        # exit()


    def get_feature(self,fn, cn):
        sample_rate = self.sampling_rate
        audio_mix, sr = librosa.load(fn, mono=False, sr=sample_rate)
        audio_tar, _ = librosa.load(cn, mono=False, sr=sample_rate)

        if audio_mix.shape[-1] < self.frame_length:
            audio_mix = np.concatenate([audio_mix, np.zeros((4, self.frame_length - audio_mix.shape[-1]))], -1)
            audio_tar = np.concatenate([audio_tar, np.zeros((4, self.frame_length - audio_tar.shape[-1]))], -1)
        elif audio_mix.shape[-1] > self.frame_length:
            # random_index = np.random.randint(audio_mix.shape[1] - self.frame_length)
            audio_mix = audio_mix[:, :self.frame_length]
            audio_tar = audio_tar[:, :self.frame_length]
        channel, length = audio_mix.shape
        shift = 1664
        frame = shift*2
        n_frame = (length - shift)//shift
        frames_input = []
        frame_delay = (frame // shift) * (shift/16000) + (frame/16000) - (shift/16000)
        
        for f in range(n_frame):
            frames_input.append(np.expand_dims(audio_mix[:,f*shift:f*shift+frame], axis=0))
        frames = np.concatenate(frames_input)  # (n_frame, 4, 3328)
        spectrogram = self.stft.compute(frames[:,0,:], sr, 512, window_time=256/16000)  # (2375, 256)
        noise_est = np.sqrt(self.estimator.compute(spectrogram).T).reshape(frames.shape[0], 25, 256)
        noise_est = np.transpose(noise_est, (0, 2, 1)) # (92, 256, 25)
        infdat = [fn, cn]

        return audio_mix, frames, audio_tar, noise_est, infdat

    def __getitem__(self, index):
        # print(self.file_list[0])
        file_name_mix = self.file_list[index]
        
        clean = '/clean_' + file_name_mix.split('_')[2] + '.wav'

        tar_name_mix = self.tar_list[index].split('/clean_')[0] + clean
        return self.get_feature(file_name_mix, tar_name_mix)


    def __len__(self):
        return len(self.file_list)
