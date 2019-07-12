import numpy as np
import random
import math
import gc
import librosa
import scipy
import matplotlib.pyplot as plt
from scipy.io import wavfile
from __future__ import division


def create_DFT(N):
    real = np.empty((N, N))
    imaginary = np.empty((N,N))
    DFT = np.zeros( (N,N), dtype = np.complex_)
    for i in range(N):
        for j in range(N):
            real[i][j] = np.cos(6.28319*i*j/N)
            imaginary[i][j] = np.sin(6.28319*i*j/N)
    #creating a complex DFT matrix for the specified N value
    for i in range(N):
        for j in range(N):
            DFT[i][j] = complex( real[i][j], -imaginary[i][j] )
    return DFT

def create_data(data, num_samples, N, overlap = 50):
    X = []
    window_start = 0
    window_end = N    
    loop = True
    frame_shift = N - N*overlap/100
    while loop == True:
        window = []
        if window_end > num_samples:
            temp = [0.0 for i in range(int(window_end - num_samples))]
            frame = np.concatenate([data[window_start:num_samples],temp])
            loop = False
        else:
            frame = data[window_start:window_end]
        frame = np.array(frame) 
        H = np.array(np.hanning(N))
        X.append(np.multiply(frame, H))
        window_start += frame_shift
        window_end += frame_shift
    X = np.array(X)
    return X

def spectrogram(data, N):
    num_samples = data.shape[0]
    DFT = create_DFT(N);
    X = create_data(data, num_samples, N)
    #spectrogram = DFT x X.T
    FX = np.dot(DFT, X.T)
    #discarding the complex conjugates (bottom half of the spectrogram)
    FX = FX[0:(N/2) + 1,:]
    rows, cols = FX.shape
    #finding the magnitude spectrogram
    magnitude = np.zeros((rows,cols))
    for i in range(rows):
        for j in range(cols):
            magnitude[i][j] = np.absolute(FX[i][j])
    return FX, magnitude

def NMF_initialization(data, latent_features):
    rows, cols = data.shape
    W = np.zeros((rows,latent_features))
    H = np.zeros((latent_features,cols))
    rows, cols = W.shape
    for i in range(rows):
        for j in range(cols):
            W[i][j] = float(random.randrange(100,500))
    rows, cols = H.shape
    for i in range(rows):
        for j in range(cols):
            H[i][j] = float(random.randrange(1,100))
    return W, H

def NMF1(data, Winitial, Hinitial, max_iterations = 1300):     
    epsilon = 1e-5
    W = Winitial.copy()
    H = Hinitial.copy()
    #iteratively apply the multiplicative update rule to W and H
    for iterations in range(max_iterations):
        Wterm = np.dot(data, H.T)/( np.dot( np.dot(W,H), H.T ) + epsilon)
        W = np.multiply(W, Wterm)
        Hterm = np.dot(W.T, data)/( np.dot( np.dot(W.T,W), H ) + epsilon)
        H = np.multiply(H, Hterm)
        #clear memory after every 400 iterations
        if iterations%400 == 0:
            gc.collect()
    return W, H

def NMF2(data, Winitial, Hinitial, max_iterations = 1300):     
    epsilon = 1e-5
    W = Winitial.copy()
    H = Hinitial.copy()
    #iteratively apply the multiplicative update rule to H only
    for iterations in range(max_iterations):
        Hterm = np.dot(W.T, data)/( np.dot( np.dot(W.T,W), H ) + epsilon)
        H = np.multiply(H, Hterm)
        #clear memory after every 400 iterations
        if iterations%400 == 0:
            gc.collect()
    return W, H


def recovery(num_samples, data, N):
    real = np.empty((N, N))
    for i in range(N):
        for j in range(N):
            real[i][j] = np.cos(6.28319*i*j/N)
    rows, cols = data.shape
    data_real = np.zeros((rows,cols))
    for i in range(rows):
        for j in range(cols):
            data_real[i][j] = data[i][j].real
    X_Inverse = np.dot(real.T,data_real)
    rows, cols = X_Inverse.shape
    X_Recovered = np.zeros((cols, num_samples))
    for i in range(cols):
        X_Recovered[i, (512 * i):((512 * i) + N)] = X_Inverse[:, i]
    X_Recovered = X_Recovered.sum(axis = 0)
    X_Recovered = X_Recovered/(N*10)
    return np.array(X_Recovered)


if __name__ == "__main__":
    #reading all signals
    trs_sampling_frequency, trs = wavfile.read('trs.wav')
    trn_sampling_frequency, trn = wavfile.read('trn.wav')
    x_sampling_frequency, x = wavfile.read('x_nmf.wav')
	
    #creating arrays for all the signals
    trs = np.array(trs, dtype = float)
    trn = np.array(trn, dtype = float)
    x = np.array(x, dtype = float)
	
    #finding the spectrograms and magnitude spectrograms for each signal
    trs_spectrogram, trs_magnitude = spectrogram(trs,N)
    trn_spectrogram, trn_magnitude = spectrogram(trn,N)
    x_spectrogram, x_magnitude = spectrogram(x,N)
	
    #finding Ws through NMF
    W, H = NMF_initialization(trs_magnitude, latent_features)
    Ws, Hs = NMF1(trs_magnitude, W, H)
    #finding Wn through NMF
    W, H = NMF_initialization(trn_magnitude, latent_features)
    Wn, Hn = NMF1(trn_magnitude, W, H)
    #Initializing Wx using Ws and Wn
    Wx = np.hstack((Ws,Wn))
    #initializing Hx with random positive numbers
    rows = Wx.shape[1]
    cols = x_magnitude.shape[1]
    Hx = np.zeros((rows,cols))
    for i in range(rows):
        for j in range(cols):
            Hx[i][j] = float(random.randrange(50,500))
    #finding Hx using the modified NMF where only H is computed
    Wx, Hx = NMF2(x_magnitude, Wx, Hx)
	
    #Estimating S
    phase = np.divide(x_spectrogram,x_magnitude)
    S_estimate = np.dot( Ws, Hx[0:30,:])
    N_estimate = np.dot( Wn, Hx[30:60,:] )
    S_estimate = np.multiply( phase, S_estimate)
    S_estimate_conjugate = S_estimate.copy()
    rows, cols = S_estimate.shape
    for i in range(rows):
        for j in range(cols):
            S_estimate_conjugate[i][j] = np.conj(S_estimate_conjugate[i][j])
    signal = np.vstack((S_estimate, S_estimate_conjugate))[0:1024,:]
    num_samples1 = trs.shape[0]
    num_samples2 = x.shape[0]
    X_recovered = recovery(num_samples1, signal, N)
    X_recovered = X_recovered[0:num_samples2]
    X_recovered = np.array(X_recovered, dtype = "int16")
	
	#recovered audio
    wavfile.write("xnmf1.wav", 16000, X_recovered)



  