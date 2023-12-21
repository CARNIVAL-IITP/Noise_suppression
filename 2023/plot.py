import librosa
import numpy as np
import librosa.display
import os
import csv
import matplotlib.pyplot as plt
import glob 


def plotspec(ref_dir, noise_di, inf_dir, inf2_dir):

    sr = 16000
    hop_length = 64
    n_fft = 1024

    # refs = os.listdir(ref_dir)
    # infs = os.listdir(inf_dir)
    # mixs = os.listdir(mix_dir)
    # refs.sort()
    # infs.sort()
    # mixs.sort()
    # print(mixs[0])

    fig = plt.figure(figsize=(15, 12))

    s1_w = fig.add_subplot(4, 2, 1)
    y, sr = librosa.load(ref_dir, sr=sr)
    y = y[:159744]
    librosa.display.waveshow(y, sr=sr)
    plt.title('Clean')

    s1 = fig.add_subplot(4, 2, 2)
    stft = librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    log_spectrogram = librosa.amplitude_to_db(magnitude, ref=np.max)
    librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')


    s2_w = fig.add_subplot(4, 2, 3)
    y_noise, sr = librosa.load(noise_dir, sr=16000)
    y = y+y_noise[:159744]
    y = y[:159744]
    librosa.display.waveshow(y, sr=sr)
    plt.title('Noisy')

    s2 = fig.add_subplot(4, 2, 4)
    stft = librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    log_spectrogram = librosa.amplitude_to_db(magnitude, ref=np.max)

    librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
    # print("Wave length: {}, Mel_S shape:{}".format(len(y) / sr, np.shape(stft)))


    s3_w = fig.add_subplot(4, 2, 5)
    y, sr = librosa.load(inf_dir, sr=16000)
    
    y = y[:159744]
    librosa.display.waveshow(y, sr=sr)
    plt.title('MIMO DCCRN')

    s3 = fig.add_subplot(4, 2, 6)
    stft = librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    log_spectrogram = librosa.amplitude_to_db(magnitude, ref=np.max)

    librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
    
    s4_w = fig.add_subplot(4, 2, 7)
    y, sr = librosa.load(inf2_dir, sr=16000)
    y = y[:159744]
    librosa.display.waveshow(y, sr=sr)
    plt.title('MIMO DCCRN_MS')

    s4 = fig.add_subplot(4, 2, 8)
    stft = librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    log_spectrogram = librosa.amplitude_to_db(magnitude, ref=np.max)

    librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
    # print("Wave length: {}, Mel_S shape:{}".format(len(y) / sr, np.shape(stft)))

    plt.tight_layout()
    name = noise_dir.split('/')[-1]
    plt.savefig(f'{name}.png')
    plt.show()
    plt.close


ref_dir = '/home/user/share/national/iitp3/mimo_dccrn_8ch/iitp_8ch/tt/clean/clean_00036.wav'

noise_dir =ref_dir.replace('speech', 'noise')


# mix_dir = glob.glob('/home/spteam/dail/DNS2020_4/tt/mix/noisy_{}*.wav'.format(ref_dir.split('/')[-1].split('_')[1][:5]))[0]
# inf_dir = '/home/spteam/dail/IITP/221025_mimo_dccrn_300ms_sitec/inference/mimodccrn_300_sitec/wav/noisy_00017_AirConditioner_4_snr20_3.94.wav'
inf = '/home/user/share/national/IITP/3th/results/inference/wav/noisy_00036_R1_linear_az0_el10_r2.1_n8_snr1_2.68.wav'
file_name = noise_dir.split('/')[-1].replace('.wav', '')
inf_dir = glob.glob(inf)[0]


# inf2 = '/home/spteam/dail/IITP/221019_mimo_dccrn_MS_300ms_adamW300/inference/mimodccrn_300_MS/wav/'
inf2_dir =inf_dir

plotspec(ref_dir, noise_dir, inf_dir, inf2_dir)

# SDRI(ref_dir, inf_dir, mix_dir)