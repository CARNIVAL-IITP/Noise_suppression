""" Yangang Cao 2021.4.24 1:22am"""

import torch
import torch.nn as nn
import torch.nn.functional as F



def pcen(x, eps=1E-6, s=0.025, alpha=0.98, delta=2, r=0.5, training=False):
    frames = x.split(1, -1)
    m_frames = []
    last_state = None
    for frame in frames:
        if last_state is None:
            last_state = s * frame
            m_frames.append(last_state)
            continue
        if training:
            m_frame = ((1 - s) * last_state).add_(s * frame)
        else:
            m_frame = (1 - s) * last_state + s * frame
        last_state = m_frame
        m_frames.append(m_frame)
    M = torch.cat(m_frames, 1)
    if training:
        p = (M + eps).pow(alpha)
        pcen_ = (x / (M + eps).pow(alpha) + delta).pow(r) - delta ** r
    else:
        pcen_ = x.div_(M.add_(eps).pow_(alpha)).add_(delta).pow_(r).sub_(delta ** r)
    return pcen_


class PCENTransform(nn.Module):

    def __init__(self, eps=1E-6, s=0.025, alpha=0.98, delta=2, r=0.5, trainable=True):
        super().__init__()
        if trainable:
            self.log_s = nn.Parameter(torch.log(torch.Tensor([s])))
            self.log_alpha = nn.Parameter(torch.log(torch.Tensor([alpha])))
            self.log_delta = nn.Parameter(torch.log(torch.Tensor([delta])))
            self.log_r = nn.Parameter(torch.log(torch.Tensor([r])))
        else:
            self.s = s
            self.alpha = alpha
            self.delta = delta
            self.r = r
        self.eps = eps
        self.trainable = trainable

    def forward(self, x):
        if self.trainable:
            x = pcen(x, self.eps, torch.exp(self.log_s), torch.exp(self.log_alpha), torch.exp(self.log_delta), torch.exp(self.log_r), self.training and self.trainable)
        else:
            x = pcen(x, self.eps, self.s, self.alpha, self.delta, self.r, self.training and self.trainable)
        x = x.unsqueeze(dim=1).permute(2, 1, 0)

        return x


class StandardConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(StandardConv1d, self).__init__()
        self.StandardConv1d = nn.Sequential(
            nn.Conv1d(in_channels = in_channels,
                    out_channels = out_channels,
                    kernel_size = kernel_size,
                    stride = stride,
                    padding = stride //2),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.StandardConv1d(x)


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DepthwiseSeparableConv1d, self).__init__()
        self.DepthwiseSeparableConv1d = nn.Sequential(
            nn.Conv1d(in_channels = in_channels,
                    out_channels = out_channels,
                    kernel_size = 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels = out_channels,
                    out_channels = out_channels,
                    kernel_size = kernel_size,
                    stride = stride,
                    padding = kernel_size // 2,
                    groups = out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace = True))

    def forward (self, x):
        return self.DepthwiseSeparableConv1d(x)

class GRUBlock(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels, bidirectional):
        super(GRUBlock, self).__init__()
        self.GRU = nn.GRU(input_size=in_channels, hidden_size=hidden_size, bidirectional=bidirectional)

        self.conv = nn.Sequential(nn.Conv1d(hidden_size * (2 if bidirectional==True else 1), out_channels, kernel_size = 1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True))

    def forward(self, x):
        output,h = self.GRU(x)
        freq, time, channel = output.shape
        # output = output.reshape(time, channel, freq)
        output = output.transpose(1,2)
        output = self.conv(output)
        return output

class FirstTrCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(FirstTrCNN, self).__init__()
        self.FirstTrCNN = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(in_channels = out_channels,
                                out_channels = out_channels,
                                kernel_size = kernel_size,
                                stride = stride,
                                padding = stride//2-1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.FirstTrCNN(x)


class TrCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(TrCNN, self).__init__()
        self.TrCNN = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(in_channels = out_channels,
                                out_channels = out_channels,
                                kernel_size = kernel_size,
                                stride = stride,
                                padding = stride//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self,x1,x2):
        diffY = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, [diffY // 2, diffY - diffY // 2, 0, 0])
        x = torch.cat((x1,x2),1)
        output = self.TrCNN(x)
        return output

class LastTrCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(LastTrCNN, self).__init__()
        self.LastTrCNN = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(in_channels = out_channels,
                                out_channels = out_channels,
                                kernel_size = kernel_size,
                                stride = stride,
                                padding=stride//2))

    def forward(self,x1,x2):
        diffY = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, [diffY // 2, diffY - diffY // 2, 0, 0])
        x = torch.cat((x1,x2),1)
        output = self.LastTrCNN(x)
        return output

class gumbel_softmax(nn.Module):
    def __init__(self, tau=1.0, hard=True):
        super(gumbel_softmax, self).__init__()
        self.tau = tau
        self.hard = hard

    def gumbel_softmax_sample(self, logits, tau, eps=1e-20):
        u = torch.rand(logits.shape)
        g = -torch.log(-torch.log(u + eps) + eps)
        x = logits + g
        return F.softmax(x / tau, dim=-1)

    def forward(self, logits):
        sh = logits.shape
        logits = logits.reshape(-1, 1)

        y = self.gumbel_softmax_sample(logits, self.tau)
        if not self.hard:
            return y

        # n_classes = y.shape[-1]
        z = torch.argmax(y, dim=-1)
        z = F.one_hot(z, 2)
        # z = (z - y).detach() + y
        p = []
        for frame in z:
            if frame[0]==1:
                p.append(1)
            else:
                p.append(-1)
        p = torch.tensor(p).reshape(sh)
        return p



class TRUNet(nn.Module):
    def __init__(self, win_len=512, win_inx=128, fft_len=512, win_type="hanning"):
        super(TRUNet, self).__init__()
        self.win_len = win_len
        self.win_inc = win_inx
        self.win_type = win_type
        self.fft_len = fft_len

        self.pcen = PCENTransform(eps=1E-6, s=0.025, alpha=0.6, delta=0.1, r=0.2, trainable=True)
        self.down1 = StandardConv1d(in_channels=1,out_channels=64,kernel_size=5,stride=2)
        self.down2 = DepthwiseSeparableConv1d(64, 128, 3, 1)
        self.down3 = DepthwiseSeparableConv1d(128, 128, 5, 2)
        self.down4 = DepthwiseSeparableConv1d(128, 128, 3, 1)
        self.down5 = DepthwiseSeparableConv1d(128, 128, 5, 2)
        self.down6 = DepthwiseSeparableConv1d(128, 128, 3, 2)
        self.FGRU = GRUBlock(128, 64, 128, bidirectional=True)

        self.TGRU = GRUBlock(128, 128, 128, bidirectional=False)
        self.up1 = FirstTrCNN(128, 128, 3, 2)
        self.up2 = TrCNN(256, 64, 5, 2)
        self.up3 = TrCNN(192, 64, 3, 1)
        self.up4 = TrCNN(192, 64, 5, 2)
        self.up5 = TrCNN(192, 64, 3, 1)
        self.up6 = LastTrCNN(128, 5, 5, 2)

        self.phi_k = nn.Sigmoid()
        self.phi_tk = nn.Sigmoid()
        self.gumbel_k = gumbel_softmax()
        self.gumbel_tk = gumbel_softmax()
        self.beta = nn.Softplus()
        layers = [self.phi_k, self.phi_tk, self.gumbel_k, self.gumbel_tk, self.beta]
        self.phm = nn.ModuleList(layers)


    def forward(self, x):

        x_p = self.pcen(x)
        x1 = self.down1(x_p)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        time, channel, freq = x6.shape
        x7 = x6.reshape(freq, time, channel)
        x8 = self.FGRU(x7)
        freq, channel, time = x8.shape

        x9 = x8.reshape(time, freq, channel)
        # x9 = x8.transpose(0,1)

        x10 = self.TGRU(x9)
        # x10 = x10.transpose(1,2)
        x11 = self.up1(x10)

        x12 = self.up2(x11[...,1:],x5)
        x13 = self.up3(x12[...,1:],x4)
        x14 = self.up4(x13[...,1:],x3)
        x15 = self.up5(x14[...,1:],x2)
        x16 = self.up6(x15[...,1:],x1)
        outs = x16.permute(1, 2, 0)

        z = []

        for idx, layer in enumerate(self.phm):

            out = layer(outs[idx])
            z.append(out)

        beta = z[-1]+1
        mask_n_mag = beta*z[0]
        mask_tn_mag = beta*z[1]
        cos = (1+mask_n_mag**2-mask_tn_mag**2)/(2*mask_n_mag)
        sin = 1-cos**2
        mask_n_ph = cos + sin*1j
        mask_n = mask_n_mag * z[2]*mask_n_ph

        y = x * mask_n


        return y

if __name__=='__main__':

    TRU = TRUNet()
    total_params = sum(p.numel() for p in TRU.parameters())
    print("total params:",total_params* 1e-6)
    # x = torch.randn(1281, 4, 257)
    x = abs(torch.randn(257, 1251))
    print("input_shape:",x.shape)
    y = TRU(x)
    print("output_shape:",y.shape) 