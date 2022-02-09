TensorFlow implementation of the Perceptual Metric for Speech Quality Evaluation (PMSQE). This metric is
computed per frame from the magnitude spectra of the reference and processed speech signal.
Inspired on the Perceptual Evaluation of the Speech Quality (PESQ) algorithm, this loss function consists of
two regularization factors which account for the symmetrical and asymmetrical distortion in the loudness domain.
See [1] for more details.

University of Granada, Signal Processing, Multimedia Transmission and Speech/Audio Technologies
(SigMAT) Group. The software is free for non-commercial use. This program comes WITHOUT ANY WARRANTY.

Files included in this package:

    - pmsqe.py: Python/TensorFlow implementation of the proposed metric.
       - perceptual_constants.py: perceptual constants used by the metric.
       - bark_matrix_Xk.mat: matlab .mat files with the matrices required for Bark-spectrum transformation (8 and 16 kHz).
    - DNN_use_example.py : Minimal functional example of DNN training using the PMSQE metric.
    - testing_examples: folder with .wav files at 8 kHz used for testing the implementation.

    - (NEW) pmsqe_torch.py: Pytorch implementation for the 16 kHz version of the PMSQE.



References:
    [1] J.M.Martin, A.M.Gomez, J.A.Gonzalez, A.M.Peinado 'A Deep Learning Loss Function based on the Perceptual
    Evaluation of the Speech Quality', IEEE Signal Processing Letters, 2018.
