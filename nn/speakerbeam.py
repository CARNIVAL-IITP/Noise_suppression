import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib
def param(nnet, Mb=True):
    """
    Return number parameters(not bytes) in nnet
    """
    neles = sum([param.nelement() for param in nnet.parameters()])
    return neles / 10**6 if Mb else neles


class cLN(nn.Module):
    def __init__(self, dimension):
        super(cLN, self).__init__()

        # self.eps = eps
        # if trainable:
        self.gain = nn.Parameter(torch.ones(1, dimension, 1))
        self.bias = nn.Parameter(torch.zeros(1, dimension, 1))
        # else:
        #     self.gain = Variable(torch.ones(1, dimension, 1), requires_grad=False)
        #     self.bias = Variable(torch.zeros(1, dimension, 1), requires_grad=False)

    def forward(self, input, eps=1e-8):
        # input size: (Batch, Freq, Time)
        # cumulative mean for each time step
        batch_size = input.size(0)
        channel = input.size(1)
        time_step = input.size(2)

        step_sum = input.sum(1)  # B, T
        step_pow_sum = input.pow(2).sum(1)  # B, T
        cum_sum = torch.cumsum(step_sum, dim=1)  # B, T
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=1)  # B, T
        device = cum_sum.device

        entry_cnt = np.arange(channel, channel * (time_step + 1), channel)
        entry_cnt = torch.from_numpy(entry_cnt).type(input.type()).to(device)
        entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum).to(device)
        # print(entry_cnt.device)
        # print(device)
        # exit()

        cum_mean = cum_sum / entry_cnt  # B, T
        cum_var = (cum_pow_sum - 2 * cum_mean * cum_sum) / entry_cnt + cum_mean.pow(2)  # B, T
        cum_std = torch.sqrt(cum_var + eps)  # B, T

        cum_mean = cum_mean.unsqueeze(1)
        cum_std = cum_std.unsqueeze(1)

        x = (input - cum_mean.expand_as(input)) / cum_std.expand_as(input)
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())

class ChannelWiseLayerNorm(nn.LayerNorm):
    """
    Channel wise layer normalization
    """

    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: N x C x T
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x C x T => N x T x C
        x = torch.transpose(x, 1, 2)
        # LN
        x = super().forward(x)
        # N x C x T => N x T x C
        x = torch.transpose(x, 1, 2)
        return x


class GlobalChannelLayerNorm(nn.Module):
    """
    Global channel layer normalization
    """

    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalChannelLayerNorm, self).__init__()
        self.eps = eps
        self.normalized_dim = dim
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.beta = nn.Parameter(torch.zeros(dim, 1))
            self.gamma = nn.Parameter(torch.ones(dim, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        """
        x: N x C x T
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x 1 x 1
        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.mean((x - mean)**2, (1, 2), keepdim=True)
        # N x T x C
        if self.elementwise_affine:
            x = self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta
        else:
            x = (x - mean) / torch.sqrt(var + self.eps)
        return x

    def extra_repr(self):
        return "{normalized_dim}, eps={eps}, " \
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)


def build_norm(norm, dim):
    """
    Build normalize layer
    LN cost more memory than BN
    """
    if norm not in ["cLN", "gLN", "BN"]:
        raise RuntimeError("Unsupported normalize layer: {}".format(norm))
    if norm == "cLN":
        return cLN(dim)
        # return ChannelWiseLayerNorm(dim, elementwise_affine=True)
    elif norm == "BN":
        return nn.BatchNorm1d(dim)
    else:
        return GlobalChannelLayerNorm(dim, elementwise_affine=True)


class Conv1D(nn.Conv1d):
    """
    1D conv in ConvTasNet
    """

    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if squeeze:
            x = torch.squeeze(x)
        return x


class ConvTrans1D(nn.ConvTranspose1d):
    """
    1D conv transpose in ConvTasNet
    """

    def __init__(self, *args, **kwargs):
        super(ConvTrans1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if squeeze:
            x = torch.squeeze(x)
        return x


class Conv1DBlock(nn.Module):
    """
    1D convolutional block:
        Conv1x1 - PReLU - Norm - DConv - PReLU - Norm - SConv
    """

    def __init__(self,
                 in_channels=256,
                 conv_channels=512,
                 kernel_size=3,
                 dilation=1,
                 norm="cLN",
                 causal=False):
        super(Conv1DBlock, self).__init__()
        # 1x1 conv
        self.conv1x1 = Conv1D(in_channels, conv_channels, 1)
        self.prelu1 = nn.PReLU()
        self.lnorm1 = build_norm(norm, conv_channels)
        dconv_pad = (dilation * (kernel_size - 1)) // 2 if not causal else (
            dilation * (kernel_size - 1))
        # depthwise conv
        self.dconv = nn.Conv1d(
            conv_channels,
            conv_channels,
            kernel_size,
            groups=conv_channels,
            padding=dconv_pad,
            dilation=dilation,
            bias=True)
        self.prelu2 = nn.PReLU()
        self.lnorm2 = build_norm(norm, conv_channels)
        # 1x1 conv cross channel
        self.sconv = nn.Conv1d(conv_channels, in_channels, 1, bias=True)
        # different padding way
        self.causal = causal
        self.dconv_pad = dconv_pad

    def forward(self, x):
        y = self.conv1x1(x)
        y = self.lnorm1(self.prelu1(y))
        y = self.dconv(y)
        if self.causal:
            y = y[:, :, :-self.dconv_pad]
        y = self.lnorm2(self.prelu2(y))
        y = self.sconv(y)
        x = x + y
        return x

class speakerbeam(nn.Module):
    def __init__(self,model_options,
                 L=40,
                 N=256,
                 X=8,
                 R=4,
                 B=256,
                 H=128,
                 P=3,
                 norm="gLN",
                 num_spks=1,
                 non_linear="sigmoid",
                 causal=False):
        super(speakerbeam, self).__init__()
        supported_nonlinear = {
            "relu": F.relu,
            "sigmoid": torch.sigmoid,
            "softmax": F.softmax
        }
        if non_linear not in supported_nonlinear:
            raise RuntimeError("Unsupported non-linear function: {}",
                               format(non_linear))
        self.non_linear_type = non_linear
        self.non_linear = supported_nonlinear[non_linear]
        # n x S => n x N x T, S = 4s*8000 = 32000
        self.encoder = Conv1D(1, N, L, stride=L // 2, bias=False, padding=0)
        # self.encoder_2 = Conv1D(1, N//4, L, stride= L // 2, bias=False, padding=0)
        # self.encoder_3 = Conv1D(1, N // 4, L, stride=L // 2, bias=False, padding=0)
        # self.encoder_4 = Conv1D(1, N // 4, L, stride=L // 2, bias=False, padding=0)
        # self.aux_encoder_1 = Conv1D(1, N//4, L, stride=L // 2, padding=0)
        # self.aux_encoder_2 = Conv1D(1, N // 4, L, stride=L // 2, padding=0)
        # self.aux_encoder_3 = Conv1D(1, N // 4, L, stride=L // 2, padding=0)
        # self.aux_encoder_4 = Conv1D(1, N // 4, L, stride=L // 2, padding=0)
        # keep T not change
        # T = int((xlen - L) / (L // 2)) + 1
        # before repeat blocks, always cLN
        # self.ln = cLN(N)
        self.ln = GlobalChannelLayerNorm(N)
        # n x N x T => n x B x T
        self.proj = Conv1D(N, B, 1)
        # repeat blocks
        # n x B x T => n x B x T
        self.repeats1 = self._build_repeats(
            R,
            X,
            in_channels=B,
            conv_channels=H,
            kernel_size=P,
            norm=norm,
            causal=causal)
        # self.repeats2 = self._build_repeats(
        #     3,
        #     X,
        #     in_channels=B,
        #     conv_channels=H,
        #     kernel_size=P,
        #     norm=norm,
        #     causal=causal)
        # self.aux_repeats = self._build_repeats(
        #     1,
        #     1,
        #     in_channels=B,
        #     conv_channels=H,
        #     kernel_size=P,
        #     norm=norm,
        #     causal=causal)
        # output 1x1 conv
        # n x B x T => n x N x T
        # NOTE: using ModuleList not python list
        # self.conv1x1_2 = torch.nn.ModuleList(
        #     [Conv1D(B, N, 1) for _ in range(num_spks)])
        # n x B x T => n x 2N x T
        self.mask = Conv1D(B, num_spks * N, 1)
        # using ConvTrans1D: n x N x T => n x 1 x To
        # To = (T - 1) * L // 2 + L
        # self.decoder_1d = ConvTrans1D(
        #     N, 1, kernel_size=L, stride=L // 2, bias=False)
        self.decoder = ConvTrans1D(
            N, 1, kernel_size=L, stride=L // 2, bias=False)
        # self.decoder_2 = ConvTrans1D(
        #     N // 4, 1, kernel_size=L, stride=L // 2, bias=False)
        # self.decoder_3 = ConvTrans1D(
        #     N // 4, 1, kernel_size=L, stride=L // 2, bias=False)
        # self.decoder_4 = ConvTrans1D(
        #     N // 4, 1, kernel_size=L, stride=L // 2, bias=False)
        # self.num_spks = 1
        self.num_spks = num_spks

    def _build_blocks(self, num_blocks, **block_kwargs):
        """
        Build Conv1D block
        """
        blocks = [
            Conv1DBlock(**block_kwargs, dilation=(2**b))
            for b in range(num_blocks)
        ]
        return nn.Sequential(*blocks)

    def _build_repeats(self, num_repeats, num_blocks, **block_kwargs):
        """
        Build Conv1D block repeats
        """
        repeats = [
            self._build_blocks(num_blocks, **block_kwargs)
            for r in range(num_repeats)
        ]
        return nn.Sequential(*repeats)

    def forward(self, x):
        x = x[0]
        # a = a[0]
        # ec = ec[0]
        # print(x.shape)
        # print(a.shape)
        # exit()
        # if x.dim() >= 3:
        #     raise RuntimeError(
        #         "{} accept 1/2D tensor as input, but got {:d}".format(
        #             self.__name__, x.dim()))
        # when inference, only one utt
        if x.dim() == 1:
            x = torch.unsqueeze(x, 0)
        # print(x.shape)
        # exit()
        # # n x 1 x S => n x N x T
        # print(x.shape)
        # x=x.transpose(0,1)
        # print(x.shape)
        w = self.encoder(x)
        # x1 = x[:, 0, :]
        # x2 = x[:, 1, :]
        # x3 = x[:, 2, :]
        # x4 = x[:, 3, :]
        # w1 = F.relu(self.encoder_1(x1))
        # w2 = F.relu(self.encoder_2(x2))
        # w3 = F.relu(self.encoder_3(x3))
        # w4 = F.relu(self.encoder_4(x4))
        # w = torch.cat((w1, w2, w3, w4), 1)
        # w = F.relu(self.encoder_1(x))
        # print(w.shape)
        # exit()
        # n x B x T
        y = self.proj(self.ln(w))
        # print(y.shape)
        # n x B x T
        y = self.repeats1(y)
        # print(y.shape)

        e = torch.chunk(self.mask(y), self.num_spks, dim=1)
        # print(len(e))
        # print(e[0].shape)
        # exit()
        # n x N x T
        if self.non_linear_type == "softmax":
            m = self.non_linear(torch.stack(e, dim=0), dim=0)
        else:
            m = self.non_linear(torch.stack(e, dim=0))
        # spks x [n x N x T]
        # print(len(m))
        # print(m[0].shape)
        # exit()
        s = [w * m[n] for n in range(self.num_spks)]
        # print(len(s))
        # exit()
        # spks x n x S
        xx = [self.decoder(x, squeeze=True) for x in s]
        # xx = [torch.stack((xdec[0], xdec[1], xdec[2], xdec[3]), dim=0)]
        # xx[0] = xx[0].reshape(1, xx[0].shape[0], xx[0].shape[1])
        # print(len(xx))
        # print(xx[0].shape)
        # exit()
        # s1, s2, s3, s4 = torch.chunk(s[0], 4, dim=1)
        # xx1 = self.decoder_1(s1)
        # xx2 = self.decoder_2(s2)
        # xx3 = self.decoder_3(s3)
        # xx4 = self.decoder_4(s4)
        # xx = [torch.cat((xx1, xx2, xx3, xx4), dim=1)]

        # xx = [self.decoder_1d(x, squeeze=True) for x in s]
        # print(len(xx))
        # print(xx[0].shape)
        # print(xx[1].shape)
        # exit()
        # ec1 = ec[:, 0, :]
        # ec2 = ec[:, 1, :]
        # ec3 = ec[:, 2, :]
        # ec4 = ec[:, 3, :]
        # ecout1 = F.relu(self.encoder_1(ec1))
        # ecout2 = F.relu(self.encoder_2(ec2))
        # ecout3 = F.relu(self.encoder_3(ec3))
        # ecout4 = F.relu(self.encoder_4(ec4))
        # ecout = torch.cat((ecout1, ecout2, ecout3, ecout4), 1)
        return xx, self.encoder, m, self.decoder
        # return [self.decoder_1d(x, squeeze=True) for x in s]


def foo_conv1d_block():
    nnet = Conv1DBlock(256, 512, 3, 20)
    print(param(nnet))


def foo_layernorm():
    C, T = 256, 20
    nnet1 = nn.LayerNorm([C, T], elementwise_affine=True)
    print(param(nnet1, Mb=False))
    nnet2 = nn.LayerNorm([C, T], elementwise_affine=False)
    print(param(nnet2, Mb=False))


def foo_conv_tas_net():
    x = torch.rand(4, 1000)
    nnet = speakerbeam(norm="cLN", causal=False)
    # print(nnet)
    print("ConvTasNet #param: {:.2f}".format(param(nnet)))
    x = nnet(x)
    s1 = x[0]
    print(s1.shape)


if __name__ == "__main__":
    model = speakerbeam(1)
    s = 0
    import numpy as np

    for param in model.parameters():
        size = np.prod(list(param.size()))
        # print(size)
        s += size
        # print(param.size())

    print(s)
    # foo_conv_tas_net()
    # foo_conv1d_block()
    # foo_layernorm()
