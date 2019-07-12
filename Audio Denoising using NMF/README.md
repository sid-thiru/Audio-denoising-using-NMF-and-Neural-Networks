# AUDIO DENOISING USING NON-NEGATIVE MATRIX FACTORIZATION

This implementation uses NMF to separate speech from noise. 'trs.wav' and 'trn.wav' are the speech and noise signals used for training the network. 
'x_nmf.wav' contains a mix of speech (same voice) and noise.

## Overview of the steps involved
* X, S and N represent the mixed signal (x_nmf), the training signal for speech (trs) and the training signal for noise (trn).
* Short time Fourier transforms are applied to S, N and X, and their magnitude spectra are found.
* NMF models are trained on S and N. From each of these NMF models, the W matrix (the features matrix of NMF) is obtained. 
* Using these two features matrices, a third NMF model is trained on X. The idea is to reuse the basis vectors learned from S and N in training a model on X.
* An estimate of S is obtained using this third NMF model, and is converted back to WAV format.
