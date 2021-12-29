from attrdict import AttrDict
from losses.loss_util import get_lossfns
from utils import AverageMeter
import argparse, data, json, nn, numpy as np, os, time, torch
import glob, librosa
from data.feature_utils import get_istft
import matplotlib
import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
import soundfile as sf
import torch.nn as nn
frm_num = 9
out_wav = torch.ones((4, 9, 16000))
frame = 16000
shift = 8000





# wav = out_wav[:, :, :shift]
#
# for k in range(frm_num - 1):
#     wav_m = out_wav[k + 1][:, :shift]
#     if k == frm_num - 2:
#         wav = torch.cat([wav, out_wav[k + 1]], -1)
#     else:
#         wav = torch.cat([wav, wav_m], -1)
#


# channel, batch, segment_size = out_wav.shape
# segment_stride = segment_size // 2
#
# input1 = torch.concat([out_wav[:, :, :segment_stride].contiguous().view(channel, -1), torch.zeros(channel, segment_stride)], dim=1)
# input2 = torch.concat([torch.zeros(channel, segment_stride), out_wav[:, :, segment_stride:].contiguous().view(channel, -1)], dim=1)
#
# output = input1 + input2
# print(output)

