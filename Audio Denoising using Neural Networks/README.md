# AUDIO DENOISING USING NEURAL NETWORKS

This implementation uses a Neural Network to separate speech from noise. 'trs.wav' and 'trn.wav' are the speech and noise signals used for training the network. 
'tex.wav' is the noisy signal on which the network is tested.

## Overview of the steps involved
* Let X = S + N, where S is the speech and N is the Noise. 'trs.wav' corresponds to S and 'trn.wav' corresponds to N
* Short time Fourier transforms are applied to S, N and X, and their magnitude spectra are found.
* An ideal binary mask is defined. This IBM matrix is used as the target variable for neural network training.
* The network has one hidden layer with tanh activation, and an output layer with sigmoid activation.
