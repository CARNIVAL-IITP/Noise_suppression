# coding=utf-8

# Minimal functional example of use of the Perceptual Metric for Speech Quality Evaluation (PMSQE). In this example we
# train a simple DNN for speech enhancement. Only a file is used for simplicity sake. Extension to proper datasets is
# trivial. See [1] for more details about PMSQE.
#
# Implemented by Juan M. Martin
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

import time

import numpy as np
import tensorflow as tf

from scipy.signal import hanning
from scipy.io import wavfile

#####################################################
# We need to import the PMSQE functions and constants
import pmsqe
#####################################################


###################################################################################
####                     Additional auxiliary functions                        ####
###################################################################################
def stft(x, w, step):
    # Short-time Fourier Transform
    #   x    : input signal vector
    #   w    : vector with the analysis window to be used
    #   step : window shift

    # Zero padding to STFT calculation
    nsampl = len(x)
    wlen = len(w)
    nframe = int(np.ceil((float(nsampl - wlen) / step))) + 1
    dif = wlen + (nframe - 1) * step - nsampl
    x = np.append(x, np.zeros(dif))

    # Zero padding in the edges
    ntotal = nsampl + dif + 2 * (wlen - step)
    x = np.append(np.zeros(wlen - step), np.append(x, np.zeros(wlen - step)))

    # DFT computation per frame (hal spectra)
    Xtf = np.array([np.fft.rfft(w * x[i:i + wlen])
                    for i in range(0, ntotal - wlen + 1, step)]) + 1e-12*(1+1j)

    return Xtf

def istft(Xtf, w, step, nsampl):
    # Inverse Short-time Fourier Transform (aka Overlap and Add)
    #   Xtf  : input matrix with STFT
    #   w    : vector with the window used during analysis
    #   step : window shift

    # Parameters
    nframe, nbin = Xtf.shape
    wlen = len(w)
    ntotal = (nframe - 1) * step + wlen

    # Overlapp-add method
    x = np.zeros(ntotal)

    ind = 0
    for i in range(0, ntotal - wlen + 1, step):
        Xt = Xtf[ind]
        x[i:i + wlen] += w * np.fft.irfft(Xt)
        ind += 1

    return x[(wlen - step):(wlen - step) + nsampl]

def random_uniform(shape):
    
    return tf.random_uniform(shape, minval=-tf.sqrt(6. / (shape[0] + shape[1])),
                                 maxval=tf.sqrt(6. / (shape[0] + shape[1])))

###################################################################################
###################################################################################



def main():
    
    # Parameters used in this example
    wlen = 256                                  # Window length
    step = 128                                  # Window shift
    hidden_layers_sizes = [128, 128, 128]       # Three hidden layers with 128 units each
    learning_rate = 1e-3                        # DNN learning rate
    epochs = 100                                # Number of epochs in DNN training

    # Clean and noisy speech example files
    file_clean = 'testing_examples/sp04.wav'
    file_noisy = 'testing_examples/sp04_babble_sn10.wav'

    # Initialize PMSQE required constants:
    #   - 8kHz mode
    #   - Use the windowing correction factor corresponding to a Squared Hann window (with shift = length/2 )
    #   - Activate equalizations corresponding to the GFEQ mode (see [1])
    pmsqe.init_constants(Fs=8000,
                         Pow_factor=pmsqe.perceptual_constants.Pow_correc_factor_SqHann,
                         apply_SLL_equalization=True,
                         apply_bark_equalization=True,
                         apply_on_degraded=True,
                         apply_degraded_gain_correction=True)

    # Analysis window for STFT
    factor_N = wlen / (step * 2)
    w = np.sqrt(hanning(wlen, False) / factor_N)    # Squared Hann window (with shift = length/2 )

    # Numpy adn Tensorflow seeds (for reproducible results)
    tf.set_random_seed(1234)
    np.random.seed(1234)



    ##### FILE READING AND PREPARATION ########
    ###########################################

    data_clean = wavfile.read(file_clean)[1] * 1.0
    data_noisy = wavfile.read(file_noisy)[1] * 1.0
    
    # STFT computation
    Ytf = stft(data_noisy, w, step)
    Xtf = stft(data_clean, w, step)
    
    # Log-magnitude spectra
    # Factor 2 of LPS is removed during normalization, so it is not included here
    Ylps = np.log(np.abs(Ytf))
    Xlps = np.log(np.abs(Xtf))
    
    # Mean and variance normalization (over the single file in this example)
    mean_vector = Ylps.mean(axis = 0)
    std_vector = Ylps.std(axis = 0)
    
    Ydata = (Ylps - mean_vector) / std_vector
    Xdata = (Xlps - mean_vector) / std_vector



    ################ DNN MODEL ################
    ###########################################

    print "- DNN architecture definition"

    # Input and output placeholders
    input_dim = wlen / 2 + 1        # DNN input layer size
    target_dim = wlen / 2 + 1       # DNN output layer size
    input_pl = tf.placeholder(tf.float32, shape = (None, input_dim), name = 'input')
    target_pl = tf.placeholder(tf.float32, shape = (None, target_dim), name = 'target')
    
    # Upper layers
    with tf.name_scope('Model'):

        previous_layer = input_pl
        previous_layer_num_units = int(previous_layer.get_shape()[1])

        # Hidden layers
        for idx, current_layer_num_units in enumerate(hidden_layers_sizes, start = 1):

            with tf.name_scope('HiddenLayer%d' % idx):

                # Hidden layers parameters
                weights = tf.Variable(
                    random_uniform([previous_layer_num_units, current_layer_num_units]), name='Weights')
                
                biases = tf.Variable(tf.zeros([current_layer_num_units]), name='Biases')

                # Hidden layer operations
                hidden_layer = tf.nn.relu(tf.add(tf.matmul(previous_layer, weights), biases))

                previous_layer = hidden_layer
                previous_layer_num_units = current_layer_num_units

        # Output layer
        output_layer_num_units = int(target_pl.get_shape()[1])

        with tf.name_scope('OutputLayer'):
            weights = tf.Variable(random_uniform([previous_layer_num_units, output_layer_num_units]),
                                  name='Weights')
            
            biases = tf.Variable(tf.zeros([output_layer_num_units]), name='Biases')

            output = tf.add(tf.matmul(previous_layer, weights), biases)

    ############### LOSS FUNCTION #############
    ###########################################
    # While MSE directly uses the LPS normalized spectra ...
    mse_loss = tf.reduce_mean(tf.square(output - target_pl), axis=1)
    # ... PMSQE requires the raw power spectra of target and output

    # First we de-normalize and undo the LPS operation:
    ref_spectra = tf.exp(2.0 * (target_pl * std_vector + mean_vector))
    deg_spectra = tf.exp(2.0 * (output * std_vector + mean_vector))

    # Then PMSQE can be computed from the (power) spectra
    pmsqe_loss = pmsqe.per_frame_PMSQE(ref_spectra, deg_spectra, alpha = 0.1)
    # Note that per_frame_PMSQE allow us to set the weight of the PMSQE distortion
    
    # Cost computation and optimization
    cost = tf.reduce_mean(mse_loss + pmsqe_loss, name = 'loss_function')
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate,
                                        name = 'adam_opt').minimize(cost)


    ############# DNN TRAINING ################
    ###########################################

    # TF variables initializer
    init = tf.global_variables_initializer()

    #saver = tf.train.Saver()

    # TF session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config = config) as sess:
        
        sess.run(init)
        print "- Training start"

        # Training epochs
        start_time = time.time()
        for epoch in range(epochs):

            # Cost computation and optimization
            _, c = sess.run([optimizer, cost], feed_dict = {input_pl: Ydata,
                                                            target_pl: Xdata})

            # Print epoch cost and summary record
            #print "   Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c)

            #saver.save(sess, network)
        total_wall_time = time.time() - start_time

        print "  *Optimization finished"
        print

        ############# DNN TESTING ################
        ###########################################

        # Parameters for ISTFT
        ly = len(data_noisy)
        Yph = np.angle(Ytf)

        # DNN processing
        X_enh = output.eval(feed_dict = {input_pl: Ydata})

        # Processing for obtaining raw audio
        # STFF coefficients
        Xlps_enh = std_vector * X_enh + mean_vector
        Xtf_enh = np.exp(Xlps_enh + 1j * Yph)

        # ISTFT
        data_enh = istft(Xtf_enh, w, step, ly)

        # Saving final speech signal
        file_out = file_noisy[:-4] + '_enhanced.wav'
        print "- Saving example output file: ", file_out
        wavfile.write(file_out, 8000, data_enh.astype('int16'))

        print "- Total time:", total_wall_time

if __name__ == '__main__':
    main()
