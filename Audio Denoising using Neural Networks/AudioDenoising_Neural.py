import numpy as np
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




#Neural network implementation

#hidden layer - tanh
def update_L1(input, weights1, weights2, bias1, X, error, delta):
    alpha = 0.00095
    gradient = 1.0 - np.tanh(input)**2
    weights_intermediate = np.dot(weights2.T, delta)
    product = np.multiply(input, weights_intermediate)
    delta_weights = np.dot(product, X.T)
    weights1 = weights1 - (alpha * delta_weights)
    bias1 = bias1 - (alpha * product)
    return weights1, bias1

#output layer - sigmoid
def update_L2(input, weights, bias, error, X):
    alpha = 0.005
    g = 1.0 / (1.0 + np.exp(-input))
    gradient = np.multiply(g, 1-g)
    product = np.multiply(error, gradient)
    delta_weights = np.dot(product, X.T)
    weights = weights - (alpha * delta_weights)
    delta_bias = np.multiply(error, gradient)
    bias = bias - (alpha * delta_bias)
    return weights, bias, product

#Neural network main function
#200 hidden units
#there is a scaling factor to compensate for the exponential growth in values of weights and bias in Layer1
#this growth leads to overflow errors on my laptop
#without this scaling factor I am able to do only ten iterations on my laptop
def NN(X, y, num_iterations = 301):
    mu = 0
    sigma = 0.001
    hidden_units = 200
    scaling = 1e110
    rows, cols = X.shape
    weights1 = np.random.normal(mu, sigma, (hidden_units, rows))
    bias1 = np.random.normal(mu, sigma, (hidden_units, cols))
    weights2 = np.random.normal(mu, sigma, (rows, hidden_units))
    bias2 = np.random.normal(mu, sigma, (rows, cols))
    for i in range(num_iterations):
        #tanh layer (hidden)
        Layer1 = np.dot(weights1, X) + bias1
        X1 = np.tanh(Layer1)
        #sigmoid layer (output)
        Layer2 = np.dot(weights2, X1) + bias2
        y_predicted = 1.0 / (1.0 + np.exp(-Layer2))
        error = y_predicted - y
        #updating both sets of weights and biases
        weights2, bias2, delta = update_L2(Layer2, weights2, bias2, error, X1)
        weights1, bias1 = update_L1(Layer1, weights1, weights2, bias1, X, error, delta)
        #scaling all values down by a factor to avoid overflow in computation
        if i%10 == 0:
            weights1 /= scaling
            bias1 /= scaling
    return weights1, bias1, weights2, bias2, y_predicted




#signal recovery
def recovery(num_samples, X):
    real = np.empty((N, N))
    for i in range(N):
        for j in range(N):
            real[i][j] = np.cos(6.28319*i*j/N)
    rows, cols = X.shape
    for i in range(rows):
        for j in range(cols):
            X[i][j] = X[i][j].real
    X_Inverse = np.dot(real.T,X)
    rows, cols = X_Inverse.shape
    X_Recovered = np.zeros((cols, num_samples + N - 3))
    for i in range(cols):
        X_Recovered[i, (512 * i):(512 * i) + N] = X_Inverse[:, i]
    X_Recovered = X_Recovered.sum(axis = 0)
    #X_Recovered *= 10
    return np.array(X_Recovered)


def SNR(s1,s2):
    numerator = np.sum(s1**2)
    denominator = np.sum((s1-s2)**2)
    SNR = numerator/denominator
    SNR = 10*np.log10(SNR)
    return SNR


#frame size
N = 1024

#reading the wav files and converting the data to numpy array format
trs_sampling_frequency, trs = wavfile.read('trs.wav')
trn_sampling_frequency, trn = wavfile.read('trn.wav')
tes_sampling_frequency, tes = wavfile.read('tes.wav')
tex_sampling_frequency, tex = wavfile.read('tex.wav')
X = trs + trn
test = tes + tex
trs = np.array(trs, dtype = 'float64')
trn = np.array(trn, dtype = 'float64')
tes = np.array(tes, dtype = 'float64')
tex = np.array(tex, dtype = 'float64')
X = np.array(X, dtype = 'float64')
test = np.array(test, dtype = 'float64')


#padding test with zeros because it is of smaller size than trs and trn
padding = np.zeros((trs.shape[0] - tex.shape[0]), dtype = 'int16')
test_padded = np.concatenate((test, padding))

#finding the spectrograms and magnitude spectrograms
trs_spectrogram, trs_magnitude = spectrogram(trs,N)
trn_spectrogram, trn_magnitude = spectrogram(trn,N)
X_spectrogram, X_magnitude = spectrogram(X,N)
test_spectrogram, test_magnitude = spectrogram(test_padded, N)

#Ideal Binary Mask
rows, cols = X_magnitude.shape
IBM_Mask = np.zeros((rows,cols))
for i in range(rows):
    for j in range(cols):
        if(trs_magnitude[i][j] > trn_magnitude[i][j]):
            IBM_Mask[i][j] = 1
        else:
            IBM_Mask[i][j] = 0

#training the neural network using X
weights1, bias1, weights2, bias2, y_predicted = NN(X_magnitude, IBM_Mask)

#appyling the trained network parameters on tex_magnitude
Layer1 = np.dot(weights1, test_magnitude) + bias1
Layer1 = np.tanh(Layer1)
Layer2 = np.dot(weights2, Layer1) + bias2
Y = 1.0 / (1.0 + np.exp(-Layer2))

#mulitplying with mask and recovering complex conjugates
Recovered = np.multiply(Y,test_spectrogram)
Recovered_conjugate = Recovered.copy()
for i in range(rows):
    for j in range(cols):
        Recovered_conjugate[i][j] = np.conj(Recovered[i][j])

#recovery
Recovered = np.vstack((Recovered, Recovered_conjugate))[0:1024,:]
Recovered = recovery(trs.shape[0], Recovered)
Recovered = Recovered[0:tex.shape[0]]/N
Recovered = np.array(Recovered, dtype = 'int16')
wavfile.write("C:\\Users\\Sid\\Documents\\MLSP\\rec1.wav", 16000, Recovered)

#SNR
snr = SNR(tes, Recovered)
print("The SNR is ",snr)

