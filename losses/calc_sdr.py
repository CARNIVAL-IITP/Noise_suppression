import os

import librosa
# from mir_eval.separation import bss_eval_sources
import numpy as np
import torch

from .pit_criterion import cal_loss

from .bss_evals import bss_eval_sources


def calc_sdr(input, pred, refer, total_cnt, total_SISNRi, f):
    mixture = input[0].reshape(1, input[0].shape[0], input[0].shape[1])
    source = refer[0].reshape(1, refer[0].shape[0], refer[0].shape[1])
    estimate_source = -pred[0].reshape(1, 1, pred[0].shape[0])
    min_length = int(np.min([mixture.shape[2], source.shape[2], estimate_source.shape[2]]))

    mixture = mixture.cpu().numpy()[:, :, 0:min_length]
    source = source.cpu().numpy()[:, :, 0:min_length]
    estimate_source = estimate_source.cpu().numpy()[:, :, 0:min_length]
    # print(mixture.shape)
    # print(source.shape)
    # print(estimate_source.shape)
    # exit()
    # mixture_lengths = mixture_lengths.cuda()
    # padded_source = source.cuda()

    # loss, max_snr, estimate_source, reorder_estimate_source = \
    #     cal_loss(padded_source, estimate_source, mixture_lengths)
    # Remove padding and flat
    # mixture = remove_pad(padded_mixture, mixture_lengths)
    # source = remove_pad(padded_source, mixture_lengths)
    # NOTE: use reorder estimate source
    # estimate_source = remove_pad(reorder_estimate_source,
    # mixture_lengths)
    # for each utterance
    for mix, src_ref, src_est in zip(mixture, source, estimate_source):
        f.write("Utt {}".format(total_cnt + 1))
        # Compute SDRi
        # if args.cal_sdr:
        # avg_SDRi = cal_SDRi(src_ref, src_est, mix)
        # total_SDRi += avg_SDRi
        # print("\tSDRi={0:.2f}".format(avg_SDRi))
    # Compute SI-SNRi
    avg_SISNRi = cal_SISNRi(src_ref, src_est, mix)
    f.write("\tSI-SNRi={0:.2f}\n".format(avg_SISNRi))
    total_SISNRi += avg_SISNRi
    total_cnt += 1
    return total_cnt, total_SISNRi
    # if args.cal_sdr:


def cal_SDRi(src_ref, src_est, mix):
    """Calculate Source-to-Distortion Ratio improvement (SDRi).
    NOTE: bss_eval_sources is very very slow.
    Args:
        src_ref: numpy.ndarray, [C, T]
        src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [T]
    Returns:
        average_SDRi
    """
    src_ref = src_ref[0,:]
    # print(src_ref.shape)
    srclen = src_ref.shape[0]
    # print(src_est.shape)
    src_est1 = src_est[0,0:srclen]
    # src_est2 = src_est[1,0:srclen]

    src_anchor = mix
    # src_anchor = np.stack([mix, mix], axis=0)
    sdr1, sir1, sar1, popt1 = bss_eval_sources(src_ref, src_est1)
    # sdr2, sir2, sar2, popt2 = bss_eval_sources(src_ref, src_est2)
    sdr0, sir0, sar0, popt0 = bss_eval_sources(src_ref, src_anchor)
    calc1 = (sdr1[0]-sdr0[0])
    # calc2 = (sdr2[0]-sdr0[0])

    # if calc1 > calc2:
    avg_SDRi = calc1
    # else:
    #     avg_SDRi = calc2
    # avg_SDRi = ((sdr[0]-sdr0[0]) + (sdr[1]-sdr0[1])) / 2
    # print("SDRi1: {0:.2f}, SDRi2: {1:.2f}".format(sdr[0]-sdr0[0], sdr[1]-sdr0[1]))
    return avg_SDRi


def cal_SISNRi(src_ref, src_est, mix):
    """Calculate Scale-Invariant Source-to-Noise Ratio improvement (SI-SNRi)
    Args:
        src_ref: numpy.ndarray, [C, T]
        src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [T]
    Returns:
        average_SISNRi
    """
    # print(src_ref.shape)
    # print(src_est.shape)
    # print(mix.shape)
    # exit()
    sisnr1 = cal_SISNR(src_ref, src_est)
    # print(sisnr1)
    # sisnr12 = cal_SISNR(src_ref[0], src_est[1])
    # if sisnr11 > sisnr12:
    #     sisnr1 = sisnr11
    # else:
    #     sisnr1 = sisnr12

    # sisnr21 = cal_SISNR(src_ref[1], src_est[0])
    # sisnr22 = cal_SISNR(src_ref[1], src_est[1])
    # if sisnr21 > sisnr22:
    #     sisnr2 = sisnr21
    # else:
    #     sisnr2 = sisnr22

    sisnr1b = cal_SISNR(src_ref, mix)
    # print(sisnr1b)
    # sisnr2b = cal_SISNR(src_ref[1], mix)
    # print("SISNR base1 {0:.2f} SISNR base2 {1:.2f}, avg {2:.2f}".format(
    #     sisnr1b, sisnr2b, (sisnr1b+sisnr2b)/2))
    # print("SISNRi1: {0:.2f}, SISNRi2: {1:.2f}".format(sisnr1, sisnr2))
    avg_SISNRi = (sisnr1 - sisnr1b)
    # avg_SISNRi = ((sisnr1 - sisnr1b) + (sisnr2 - sisnr2b)) / 2
    return avg_SISNRi


def cal_SISNR(ref_sig, out_sig, eps=1e-8):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        ref_sig: numpy.ndarray, [T]
        out_sig: numpy.ndarray, [T]
    Returns:
        SISNR
    """
    # print(ref_sig.shape)
    # print(out_sig.shape)
    # exit()
    assert len(ref_sig) == len(out_sig)
    ref_sig = ref_sig - np.mean(ref_sig)
    out_sig = out_sig - np.mean(out_sig)
    ref_energy = np.sum(ref_sig ** 2) + eps
    proj = np.sum(ref_sig * out_sig) * ref_sig / ref_energy
    noise = out_sig - proj
    ratio = np.sum(proj ** 2) / (np.sum(noise ** 2) + eps)
    sisnr = 10 * np.log(ratio + eps) / np.log(10.0)
    return sisnr


# if __name__ == '__main__':
#     args = parser.parse_args()
#     print(args)
#     evaluate(args)

