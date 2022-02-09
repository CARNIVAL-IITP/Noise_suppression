# coding=utf-8

# TensorFlow implementation of the Perceptual Metric for Speech Quality Evaluation (PMSQE). This metric is
# computed per frame from the magnitude spectra of the reference and processed speech signal.
# Inspired on the Perceptual Evaluation of the Speech Quality (PESQ) algorithm, this loss function consists of
# two regularization factors which account for the symmetrical and asymmetrical distortion in the loudness domain.
# See [1] for more details.
#
# Implemented by Angel M. Gómez
# Tested and revised by Juan M. Martin
#
#   References:
#    [1] J.M.Martin, A.M.Gomez, J.A.Gonzalez, A.M.Peinado 'A Deep Learning Loss Function based on the Perceptual
#    Evaluation of the Speech Quality', IEEE Signal Processing Letters, 2018.
#
#
# Copyright 2018: University of Granada, Signal Processing, Multimedia Transmission and Speech/Audio Technologies
# (SigMAT) Group. The software is free for non-commercial use. This program comes WITHOUT ANY WARRANTY.
#

import tensorflow as tf
import numpy as np
import scipy.io
import scipy.signal
import perceptual_constants

# Compensations to be applied (set in init_constants function)
SLL_equalization = True
bark_equalization = True
on_degraded = False
degraded_gain_correction = True

# Perceptual constants, initially set to null, use init_constants to properly set them according to sampling frequency
Sp = None
Sl = None
# Number of bark bands
Nb = None
# Number of FFT bands (half spectra)
Nf = None
# STFTT Windowing correction factor
Pow_correc_factor = None
# Hann (periodic) window
Hann_Window = None
# Absolute listening threshold vector
abs_thresh_power = None
# Modified Zwicker's law power vector
modified_zwicker_power = None
# Vector with bark bands widths
width_of_band_bark = None
# Total bark band width
sqrt_total_width = 0
# Bark domain transformation matrix
bark_matrix = None

def overlapped_windowing(tensor, window, shift):
    """
    * General purpose function *
    This function receives a 2D tensor with secuences (dim 1) of samples (dim 0) and decomposes it in a 3D tensor of
    sequences (dim 2) of frames (dim 1) of samples (dim 0) applying windowing and overlaving over each frame.
    Note that this function can also process only one sequence (1D input) providing it decomposition in frames (2D output)

    :param tensor:    2D (1D) matrix of sequences of samples to be decomposed (inner-most dimension)
    :param window:    1D Vector definning the window to be applied in each frame (null size vectors are not allowed)
    :param shift:     Shifting applied during windowing to obtain each frame (stride)

    :return:         3D (2D) tensor with sequences, frames and samples
    """
    window_size = window.shape[0]
    overlapped_slices = tf.squeeze(tf.extract_image_patches(tensor[:, :, None, None], ksizes=[1, window_size, 1, 1],
                                                            strides=[1, shift, 1, 1], rates=[1, 1, 1, 1],
                                                            padding='VALID'))
    return overlapped_slices * window


def squared_magnitude_computation(tensor):
    """
    Computes the squared magnitude of the spectra from a tensor comprised of time domain frames.
    Input tensor can have an arbitrary shape, spectra is computed over the inner-most dimension.
    Note that a real signal is assumed so only half of the spectra is computed (faster).

    :param tensor:    Input tensor with time-domain samples

    :return:          Squared spectra (same shape as input)
    """
    spectrum = tf.square(tf.abs(tf.spectral.rfft(tensor)))
    return spectrum


def magnitude_at_standard_listening_level(spectra):
    """
    Normalizes the spectra to a standard reference listening level. Gains are based on the energy of the band-pass-filtered
    (350-3250 Hz) speech. In PESQ this is performed in the time domain. Here we take advance of the Parseval's
    theorem to approximate it in the FFT domain, HOWEVER:
       1. FFT has been computed over a WINDOWED signal (corrected by Pow_correc_factor)
       2. 350-3250 Hz are approached by indexes  11-104 (343.75-3250 Hz) at both 8 kHz and 16 kHz (256 and 512 FFT points)
       3. We only have half of spectra (non-symmetric) which makes things harder. We exploit previous condition and assume
          first and last components of spectra are null (i.e. band pass signal) so the rest ones are just duplicated,
          which simplifies a bit the computation.
    Total energy is given as sum of the energies of each spectrum (squared sum of magnitude divided by number of bands (*)).
    Pow is computed as the total energy divided by the number of samples, i.e. number of frames x Nf
    -> Mean is used here to avoid knowing matrix sizes

    (*) Only half (non-symmetric) spectra are available, so the number of bands is given by Nf/2+1. A correction factor of
    (N/2+1)/(N/2) is applied over the averaged pow (note that this is only valid for bandpass signals as here)

    Further details of this can be found in magnitude_at_standard_listening_level_numpy function and in the support documentation
    """

    # Condensed and Fs independent derivation of pow formula
    mask = np.zeros(shape=[Nf/2+1], dtype=np.float32)
    mask[11]=0.5*25.0/31.25      # 11th bin represents freq. 343.75 but we need 350 (proportional value)
    mask[12:104]=1.0
    mask[104]=0.5
    # Corrections are applied in the mask for efficiency reasons (mask is set as a TensorFlow constant!)
    mask=mask*Pow_correc_factor*(Nf+2.0)/(Nf*Nf)

    Pow = tf.reduce_mean(tf.multiply(spectra, tf.constant(mask, dtype=tf.float32)))
    return spectra*(10000000.0 / Pow)


def magnitude_at_standard_listening_level_numpy(spectra):
    """
    Performs the same function that "magnitude_at_standard_listening_level" but using numpy primitives and following a more
    didactic implementation.
    """
    mask = np.zeros(shape=[Nf/2+1], dtype=np.float32)
    mask[11]=0.5*25.0/31.25      # 11th bin represents freq. 343.75 but we need 350 (proportional value)
    mask[12:104]=1.0
    mask[104]=0.5
    # Index frequency values can be checked in this vector:
    frecs=np.linspace(0, 8000, 257)

    # Intelligible but non-general implementation:
    # Pow_correc_factor*(np.sum(np.multiply(spectra, mask)) * 1/(Nf/2+1) *(Nf/2+1)/(Nf/2) ) / (spectra.shape[0]*Nf)
    #                     /------------------------------/               /-----------/       /--------------------/
    #  /-------------/  half spectra (HS) magnitude in band    /-----/   correction for        equivalent number
    #  Correction for windowing                               averaging   only averaging           of samples
    #                                                                      half spectra
    # After simplifying:
    Pow = Pow_correc_factor*np.mean(np.multiply(spectra, mask)) * (Nf+2.0)/(Nf*Nf)
    # Alternatively:
    # Pow = sum (Energy_per_frame) / (n. total samples) = sum (Energy_per_frame) / (Nf * number of frames) = average(Energy_per_frame)/Nf
    # Energy_per_frame = average(Spectra^2) = average (Half_Spectra^2) * (Nf/2 + 1)/ (Nf/2)   (note that Half_Spectra has Nf/2+1 bands)
    # (Nf/2 + 1)/ (Nf/2) acts a correction factor due to averaging Half_Spectra instead of Spectra (only valid for band pass signals)
    # Therefore:
    # Pow = average( average(Half_Spectra^2) * (Nf/2 + 1)/(Nf/2) ))/Nf  = average(average(Half_Spectra^2)) * (Nf+2.0)/(Nf*Nf)
    # As each Spectra is obtained after windowing the signal we need to correct the loss of pow:
    # Pow = Pow * Pow_correc_factor
    # See support documentation for further details
    return spectra*(10000000.0 / Pow)


def bark_computation(spectra):
    """
    Bark spectrum estimation. No spreading function is applied.
    Only 129- and 257-band spectra (8 and 16 Khz) are allowed, providing 42- and 49-band Bark spectra.

    :param spectra:    2D matrix with square magnitude spectra (inner dimension)

    :return:           2D matrix with Bark spectra
    """
    bark_spectra = tf.matmul(spectra, bark_matrix)*Sp
    return bark_spectra


def compute_audible_power(bark_spectra, factor=1.0):
    """
    The audible power in the Bark domain includes only Bark components in the power calculation larger than the absolute
    hearing threshold of each band, i.e. only components which are audible. This function computes the audible power of
    each frame according to a factor times the absolute hearing threshold (1.0 by default).

    :param spectra:    2D matrix with bark spectra (inner dimension)
    :param factor:     optional scalar value to be applied to the hearing threshold

    :return:           1D vector with audible power in each frame
    """
    cond = tf.greater(bark_spectra, abs_thresh_power*factor)

    return tf.reduce_sum(tf.where(cond,bark_spectra,tf.zeros(tf.shape(bark_spectra))),axis=1)


def bark_gain_equalization(bark_spectra_ref, bark_spectra_deg):
    """
    To compensate for short-term gain variations, the ratio between the audible power of the original and the degraded
    signal is computed and the latter one compensated. This ratio is bounded to the range [3.0e-4, 5].
    Note that original PESQ applies a filtering across time for gain estimation. This filtering is overridden here due
    to practical reasons.

    :param bark_spectra_ref:    2D matrix with the reference signal's bark spectra (inner dimension)
    :param bark_spectra_deg:    2D matrix with the degraded signal's bark spectra (inner dimension)

    :return:                    2D matrix with the degraded signal's bark spectra corrected

    """
    audible_power_ref=compute_audible_power(bark_spectra_ref, 1.0)
    audible_power_deg=compute_audible_power(bark_spectra_deg, 1.0)
    gain = tf.div(audible_power_ref + 5.0e3, audible_power_deg + 5.0e3)
    limited_gain=tf.maximum(tf.minimum(gain, 5.0),3.0e-4)

    return tf.multiply(bark_spectra_deg,tf.expand_dims(limited_gain,-1))


def bark_frequency_equalization(bark_spectra_ref, bark_spectra_deg):
    """
    To compensate for filtering effects, a factor is computed based on the ratio of averaged (over speech active frames)
    degraded Bark spectrum to the original Bark spectrum. The original Bark spectrum is then multiplied by this factor,
    which is limited to the range of [-20 dB, 20dB].

    :param bark_spectra_ref:    2D matrix with the reference signal's bark spectra (inner dimension)
    :param bark_spectra_deg:    2D matrix with the degraded signal's bark spectra (inner dimension)

    :return:                    2D matrix with the reference signal's bark spectra corrected
    """
    # Identification of speech active frames (over reference bark spectra):
    audible_powerX100=compute_audible_power(bark_spectra_ref, 100.0)
    not_silent=tf.greater_equal(audible_powerX100, 1.0e7)

    # Threshold for active bark bins (if value is lower than 100 times audible threshold then it is set to 0)
    ref_thresholded = tf.where(tf.greater_equal(bark_spectra_ref, abs_thresh_power*100.0), bark_spectra_ref, tf.zeros(tf.shape(bark_spectra_ref)))
    deg_thresholded = tf.where(tf.greater_equal(bark_spectra_deg, abs_thresh_power*100.0), bark_spectra_deg, tf.zeros(tf.shape(bark_spectra_deg)))

    # Average Pow per bark bin in reference and degraded signals
    avg_pow_per_bark_ref = tf.reduce_sum(tf.where(not_silent, ref_thresholded, tf.zeros(tf.shape(bark_spectra_ref))),axis=0)
    avg_pow_per_bark_deg = tf.reduce_sum(tf.where(not_silent, deg_thresholded, tf.zeros(tf.shape(bark_spectra_deg))),axis=0)

    equalizer = (avg_pow_per_bark_deg+1000.0)/(avg_pow_per_bark_ref+1000.0)
    equalizer = tf.maximum(tf.minimum(equalizer, 100.0), 0.01)

    return tf.multiply(bark_spectra_ref,equalizer)


def bark_frequency_equalization_on_degraded(bark_spectra_ref, bark_spectra_deg):
    """
    To compensate for filtering effects, a factor is computed based on the ratio of averaged (over speech active frames)
    degraded Bark spectrum to the original Bark spectrum. The original Bark spectrum is then multiplied by this factor,
    which is limited to the range of [-20 dB, 20dB].

    :param bark_spectra_ref:    2D matrix with the reference signal's bark spectra (inner dimension)
    :param bark_spectra_deg:    2D matrix with the degraded signal's bark spectra (inner dimension)

    :return:                    2D matrix with the reference signal's bark spectra corrected
    """
    # Identification of speech active frames (over reference bark spectra):
    audible_powerX100=compute_audible_power(bark_spectra_ref, 100.0)
    not_silent=tf.greater_equal(audible_powerX100, 1.0e7)

    # Threshold for active bark bins (if value is lower than 100 times audible threshold then it is set to 0)
    ref_thresholded = tf.where(tf.greater_equal(bark_spectra_ref, abs_thresh_power*100.0), bark_spectra_ref, tf.zeros(tf.shape(bark_spectra_ref)))
    deg_thresholded = tf.where(tf.greater_equal(bark_spectra_ref, abs_thresh_power*100.0), bark_spectra_deg, tf.zeros(tf.shape(bark_spectra_deg)))

    # Average Pow per bark bin in reference and degraded signals
    avg_pow_per_bark_ref = tf.reduce_sum(tf.where(not_silent, ref_thresholded, tf.zeros(tf.shape(bark_spectra_ref))),axis=0)
    avg_pow_per_bark_deg = tf.reduce_sum(tf.where(not_silent, deg_thresholded, tf.zeros(tf.shape(bark_spectra_deg))),axis=0)


    equalizer = (avg_pow_per_bark_ref+1000.0)/(avg_pow_per_bark_deg+1000.0)
    equalizer = tf.maximum(tf.minimum(equalizer, 100.0), 0.01)

    return tf.multiply(bark_spectra_deg,equalizer)


def loudness_computation(bark_spectra):
    """
    Bark spectra are transformed to a sone loudness scale using Zwicker's law.

    :param bark_spectra:    2D matrix with equalized bark spectra

    :return:                2D matrix with loudness densities
    """
    loudness_dens = Sl*tf.multiply(tf.pow(abs_thresh_power / 0.5, modified_zwicker_power), tf.pow(0.5+0.5*bark_spectra / abs_thresh_power,modified_zwicker_power) -1.0 )

    cond = tf.less(bark_spectra, abs_thresh_power)
    loudness_dens_limited = tf.where(cond, tf.zeros(tf.shape(loudness_dens)), loudness_dens)

    return loudness_dens_limited


def compute_distortion_tensors(bark_spectra_ref, bark_spectra_deg):
    """
    Computes the Symmetric and the Asymmetric disturbance matrices between the reference and degraded bark spectra.
    Input Bark spectra MUST BE previously compensated.

    :param bark_spectra_ref:     2D matrix with the reference signal's bark spectra (inner dimension)
    :param bark_spectra_deg:     2D matrix with the degraded signal's bark spectra (inner dimension)

    :return:                     2D matrices with Symmetric and the Asymmetric disturbance
    """

    # After bark spectra are compensated, these are transformed to sone loudness
    original_loudness = loudness_computation(bark_spectra_ref)
    distorted_loudness = loudness_computation(bark_spectra_deg)

    # Loudness difference
    r=tf.subtract(distorted_loudness,original_loudness)
    # Masking effect computation
    m=0.25*tf.minimum(original_loudness,distorted_loudness)
    # Center clipping using masking effect
    D=tf.maximum(tf.abs(r)-m, 0)

    # Asymmetry factor computation
    Asym=tf.pow(tf.div(bark_spectra_deg + 50.0, bark_spectra_ref + 50.0), 1.2)
    cond = tf.less(Asym, 3.0)
    AF=tf.where(cond, tf.zeros(tf.shape(Asym)), tf.minimum(Asym, 12.0))
    # Asymmetric Disturbance matrix computation
    DA = tf.multiply(AF, D)

    return D, DA


def per_frame_distortion(D, DA, total_power_ref):
    """
    Computes the Symmetric and the Asymmetric disturbance per-frame metrics.

    :param D:               Symmetric disturbance matrix
    :param DA:              Asymmetric disturbance matrix
    :param total_power_ref: Audible power (per frame) vector from the reference signal

    :return:                Two 1D vectors with the Symmetric and the Asymmetric disturbance per frame
    """

    # Computation of the norms over bark bands for each frame (2 and 1 for D and DA, respectively)
    D_frame = tf.sqrt(tf.reduce_sum(tf.square(tf.multiply(D, width_of_band_bark)), axis=1))*sqrt_total_width
    DA_frame = tf.reduce_sum(tf.multiply(DA, width_of_band_bark), axis=1)

    # Weighting by the audible power raised to 0.04
    weights = tf.pow((total_power_ref + 1e5)/1e7, 0.04)

    # Bounded computation of the per frame distortion metric
    wD_frame = tf.minimum(tf.div(D_frame, weights), 45.0)
    wDA_frame = tf.minimum(tf.div(DA_frame, weights), 45.0)

    return wD_frame, wDA_frame


def per_frame_PMSQE(ref_spectra, deg_spectra, alpha = 0.1):
    """
    Defines the complete pipeline to compute a PMSQE metric per frame given a reference and a degraded spectra.
    Input spectra are assumed to be half-spectra (i.e. real signal) with 129 or 257 bands (depending on 8 and 16 Khz)

    :param ref_spectra:               2D matrix with reference spectra
    :param deg_spectra:               2D matrix with degraded spectra

    :return:                          1D vector with a 'PESQ' score per frame
    """

    if SLL_equalization:
        ref_eq_spectra = magnitude_at_standard_listening_level(ref_spectra)
        deg_eq_spectra = magnitude_at_standard_listening_level(deg_spectra)
    else:
        ref_eq_spectra = ref_spectra
        deg_eq_spectra = deg_spectra

    # Bark spectra computation:
    ref_bark_spectra = bark_computation(ref_eq_spectra)
    deg_bark_spectra = bark_computation(deg_eq_spectra)

    # Bounded equalization of the reference (or degraded) Bark spectra:
    if bark_equalization:
        if on_degraded:
           deg_bark_spectra = bark_frequency_equalization_on_degraded(ref_bark_spectra, deg_bark_spectra)
           ref_bark_spectra_corrected = ref_bark_spectra
        else:
           ref_bark_spectra_corrected = bark_frequency_equalization(ref_bark_spectra, deg_bark_spectra)
    else:
        ref_bark_spectra_corrected = ref_bark_spectra

    # Bounded gain correction of the degraded Bark spectra:
    if degraded_gain_correction:
        deg_bark_spectra_corrected = bark_gain_equalization(ref_bark_spectra_corrected, deg_bark_spectra)
    else:
        deg_bark_spectra_corrected = deg_bark_spectra

    # Distortion matrix computation from Bark spectra:
    D, DA = compute_distortion_tensors(ref_bark_spectra_corrected, deg_bark_spectra_corrected)
    # Per-frame distortion aggregation
    audible_power_ref=compute_audible_power(ref_bark_spectra_corrected, 1.0)
    wD_frame, wDA_frame = per_frame_distortion(D,DA, audible_power_ref)

    # 'PMSQE' metric per-frame
    return alpha*(wD_frame + 0.309*wDA_frame)


def init_constants(Fs = 8000, Pow_factor = perceptual_constants.Pow_correc_factor_Hann, apply_SLL_equalization=True,
                   apply_bark_equalization=True, apply_on_degraded=True, apply_degraded_gain_correction=True):
    """
    Initialization of perceptual constants which depends on the sampling frequency (i.e 8 or 16 kHz).
    It is MANDATORY calling this function before any other.

    :param Fs:     Sampling frequency in Hz (default 8000)
    :param Pow_factor:                      Power factor to compensate the windowing used in your FFT (see perceptual_constants.py)
    :param apply_SLL_equalization:          Equalize the level of both signals to a standard listening level (SLL)
    :param apply_bark_equalization:         Equalize the reference signal to compensate filtering effects
    :param apply_on_degraded:               Equalize the degraded signal instead
    :param apply_degraded_gain_correction:  Compensate small gain variations in the degraded signal

    """
    global SLL_equalization
    global bark_equalization
    global on_degraded
    global degraded_gain_correction

    global Sp
    global Sl
    global Nb
    global Nf
    global Pow_correc_factor
    global Hann_Window
    global abs_thresh_power
    global modified_zwicker_power
    global width_of_band_bark
    global sqrt_total_width
    global bark_matrix

    SLL_equalization = apply_SLL_equalization
    bark_equalization = apply_bark_equalization
    on_degraded = apply_on_degraded
    degraded_gain_correction = apply_degraded_gain_correction

    Pow_correc_factor = Pow_factor

    if Fs==16000:
        # Scalars
        Sp=perceptual_constants.Sp_16k
        Sl=perceptual_constants.Sl_16k
        Nb=perceptual_constants.Nb_16k
        Nf=perceptual_constants.Nf_16k
        #Vectors
        Hann_Window=tf.constant(scipy.signal.hann(Nf, False), dtype=tf.float32)
        abs_thresh_power=tf.constant(perceptual_constants.abs_thresh_power_16k, dtype=tf.float32)
        modified_zwicker_power = tf.constant(perceptual_constants.modified_zwicker_power_16k, dtype=tf.float32)
        width_of_band_bark = tf.constant(perceptual_constants.width_of_band_bark_16k, dtype=tf.float32)
        sqrt_total_width = tf.constant(np.sqrt(np.sum(perceptual_constants.width_of_band_bark_16k)), dtype=tf.float32)
        # Matrices
        mat = scipy.io.loadmat('bark_matrix_16k.mat')
        bark_matrix=tf.constant(mat["Bark_matrix_16k"], dtype=tf.float32)

    else:
        # Scalars
        Sp=perceptual_constants.Sp_8k
        Sl=perceptual_constants.Sl_8k
        Nb=perceptual_constants.Nb_8k
        Nf=perceptual_constants.Nf_8k
        #Vectors
        Hann_Window=tf.constant(scipy.signal.hann(Nf, False), dtype=tf.float32)
        abs_thresh_power=tf.constant(perceptual_constants.abs_thresh_power_8k, dtype=tf.float32)
        modified_zwicker_power = tf.constant(perceptual_constants.modified_zwicker_power_8k, dtype=tf.float32)
        width_of_band_bark = tf.constant(perceptual_constants.width_of_band_bark_8k, dtype=tf.float32)
        sqrt_total_width = tf.constant(np.sqrt(np.sum(perceptual_constants.width_of_band_bark_8k)), dtype=tf.float32)
        # Matrices
        mat = scipy.io.loadmat('bark_matrix_8k.mat')
        bark_matrix=tf.constant(mat["Bark_matrix_8k"], dtype=tf.float32)
    pass


def test_PMSQE(reference_file, distorted_file):
    """
    Example use of per_frame_PMSQE function with 8 or 16 Khz files

    :param reference_file:     Reference wav filename
    :param distorted_file:     Distorted wav filename
    """

    import scipy.io.wavfile
    import matplotlib.pyplot as plt

    # Load signals
    Fs, reference = scipy.io.wavfile.read(reference_file)
    _, degraded = scipy.io.wavfile.read(distorted_file)

    # Init PMSQE required constants
    init_constants(Fs, Pow_factor=perceptual_constants.Pow_correc_factor_Hann, apply_SLL_equalization=True,
                   apply_bark_equalization=True, apply_on_degraded=True, apply_degraded_gain_correction=True)

    # Prepare signals (windowing, STFFT)
    signals = tf.placeholder(tf.float32, [2, None])
    windowed_signals = overlapped_windowing(signals, window=Hann_Window, shift=Nf / 2)
    ref_spectra = squared_magnitude_computation(windowed_signals[0])
    deg_spectra = squared_magnitude_computation(windowed_signals[1])

    # PMSQE computation
    PMSQE_per_frame = per_frame_PMSQE(ref_spectra, deg_spectra, alpha=0.1)

    with tf.Session():
        pmsqe = PMSQE_per_frame.eval({signals: [reference, degraded]})
    print 'Average PMSQE distortion: {0:.2f} '.format(np.mean(pmsqe))
    plt.plot(np.arange(0, pmsqe.shape[0], 1), pmsqe)
    plt.title('PMSQE distortion per frame')
    plt.show()
    pass


if __name__ == "__main__":
    #print(tf.__version__)
    # Tested on tensorflow version 1.10.0
    test_PMSQE('testing_examples/sp04.wav', 'testing_examples/sp04_babble_sn10.wav')
