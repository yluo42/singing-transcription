__author__ = 'Roy'

import os
import numpy as np
from scipy.io.wavfile import read
import math
from scipy.fftpack import fft
from scipy.signal import get_window

INT16_FAC = (2**15)-1
INT32_FAC = (2**31)-1
INT64_FAC = (2**63)-1
norm_fact = {'int16': INT16_FAC, 'int32': INT32_FAC, 'int64': INT64_FAC, 'float32': 1.0 , 'float64': 1.0}


def isPower2(num):
    """
    Check if num is power of two
    """
    return ((num & (num - 1)) == 0) and num > 0


def wavread(filename):
    """
    Read a sound file and convert it to a normalized floating point array
    filename: name of file to read
    returns fs: samplint rate of file, x: floating point array
    """

    if not os.path.isfile(filename):                  # raise error if wrong input file
        raise ValueError("Input file is wrong")

    fs, x = read(filename)

    x = np.float32(x) / norm_fact[x.dtype.name]
    return fs, x.transpose()[0]


def DFT(x, w, N):
    """
    Analysis of a signal using the discrete Fourier transform
    x: input signal, w: analysis window, N: FFT size
    returns mX, pX: magnitude and phase spectrum
    """

    if not isPower2(N):                                 # raise error if N not a power of two
        raise ValueError("FFT size (N) is not a power of 2")

    if w.size > N:                                        # raise error if window size bigger than fft size
        raise ValueError("Window size (M) is bigger than FFT size")

    hN = (N/2)+1                                            # size of positive spectrum, it includes sample 0
    hM1 = int(math.floor((w.size+1)/2))                     # half analysis window size by rounding
    hM2 = int(math.floor(w.size/2))                         # half analysis window size by floor
    fftbuffer = np.zeros(N)                                 # initialize buffer for FFT
    w = w / sum(w)                                          # normalize analysis window
    xw = x*w                                                # window the input sound
    fftbuffer[:hM1] = xw[hM2:]                              # zero-phase window in fftbuffer
    fftbuffer[-hM2:] = xw[:hM2]
    X = fft(fftbuffer)                                      # compute FFT
    absX = abs(X[:hN])                                      # compute ansolute value of positive side
    absX[absX < np.finfo(float).eps] = np.finfo(float).eps    # if zeros add epsilon to handle log
    mX = 20 * np.log10(absX)                                # magnitude spectrum of positive frequencies in dB
    pX = np.unwrap(np.angle(X[:hN]))                        # unwrapped phase spectrum of positive frequencies
    return mX, pX


def STFT(x, w='hanning', N=32, H=8) :
    """
    Analysis of a sound using the short-time Fourier transform
    x: input array sound, w: analysis window, N: FFT size, H: hop size
    returns xmX, xpX: magnitude and phase spectra
    """
    if H <= 0:                                   # raise error if hop size 0 or negative
        raise ValueError("Hop size (H) smaller or equal to 0")

    w = get_window(w, 2*H+1)                          # get analysis window
    M = w.size                                      # size of analysis window
    hM1 = int(math.floor((M+1)/2))                  # half analysis window size by rounding
    hM2 = int(math.floor(M/2))                      # half analysis window size by floor
    x = np.append(np.zeros(hM2), x)                  # add zeros at beginning to center first window at sample 0
    x = np.append(x, np.zeros(hM2))                  # add zeros at the end to analyze last sample
    pin = hM1                                       # initialize sound pointer in middle of analysis window
    pend = x.size-hM1                               # last sample to start a frame
    w = w / sum(w)                                  # normalize analysis window
    y = np.zeros(x.size)                            # initialize output array
    while pin <= pend:                                # while sound pointer is smaller than last sample
        x1 = x[pin-hM1:pin+hM2]                       # select one frame of input sound
        mX, pX = DFT(x1, w, N)                      # compute dft
        if pin == hM1:                                # if first frame create output arrays
            xmX = np.array([mX])
            xpX = np.array([pX])
        else:                                         # append output to existing array
            xmX = np.vstack((xmX, np.array([mX])))
            xpX = np.vstack((xpX, np.array([pX])))
        pin += H                                      # advance sound pointer
    return xmX, xpX