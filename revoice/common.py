import numpy as np
import scipy.io.wavfile as wavfile
import scipy.interpolate as ipl
import scipy.signal as sp
import scipy.special as spec
import numba as nb

windowDict = {
    #           func(N), main-lobe-width, mean
    'hanning': (sp.hanning, 1.5, 0.5),
    'blackman': (sp.blackman, 1.73, 0.42),
    'blackmanharris': (sp.blackmanharris, 2.0044, (35875 - 3504 * np.pi) / 100000),
}

def loadWav(filename): # -> samprate, wave in float64
    samprate, w = wavfile.read(filename)
    if(w.dtype == np.int8):
        w = w.astype(np.float64) / 127.0
    elif(w.dtype == np.short):
        w = w.astype(np.float64) / 32767.0
    elif(w.dtype == np.int32):
        w = w.astype(np.float64) / 2147483647.0
    elif(w.dtype == np.float32):
        w = w.astype(np.float64)
    elif(w.dtype == np.float64):
        pass
    else:
        raise ValueError("Unsupported sample format: %s" % (str(w.dtype)))
    return w, samprate

def saveWav(filename, data, samprate):
    wavfile.write(filename, int(samprate), data)

def simpleDCRemove(x):
    return x - np.mean(x)

def _sumGaussian(x, stdev):
    return np.sqrt(np.pi) * stdev * spec.erf(x / np.sqrt(2) / stdev) / np.sqrt(2)

def sumGaussian(n, stdev):
    return _sumGaussian(n - 1, stdev) - _sumGaussian(1 - n, stdev)

def _sumGaussianSquare(x, stdev):
    return np.sqrt(np.pi) * stdev * spec.erf(x / stdev) / 2.0

def sumGaussianSquare(n, stdev):
    return _sumGaussianSquare(n - 1, stdev) - _sumGaussianSquare(1 - n, stdev)

@nb.jit(nb.types.Tuple((nb.int64, nb.int64, nb.int64, nb.int64))(nb.int64, nb.int64, nb.int64), nopython = True, cache = True)
def getFrameRange(inputLen, center, size):
    leftSize = size // 2
    rightSize = size - leftSize # for odd size

    inputBegin = min(inputLen, max(center - leftSize, 0))
    inputEnd = max(0, min(center + rightSize, inputLen))

    outBegin = max(leftSize - center, 0)
    outEnd = outBegin + (inputEnd - inputBegin)

    return outBegin, outEnd, inputBegin, inputEnd

@nb.jit(nb.float64[:](nb.float64[:], nb.int64, nb.int64), nopython = True, cache = True)
def getFrame(input, center, size):
    out = np.zeros(size, input.dtype)

    outBegin, outEnd, inputBegin, inputEnd = getFrameRange(len(input), center, size)

    out[outBegin:outEnd] = input[inputBegin:inputEnd]
    return out

@nb.jit(nb.int64(nb.int64, nb.int64), nopython = True, cache = True)
def getNFrame(inputSize, hopSize):
    return inputSize // hopSize + 1 if(inputSize % hopSize != 0) else inputSize // hopSize

def getWindow(window):
    if(type(window) is str):
        return windowDict[window]
    elif(type(window) is tuple):
        assert(len(window) == 3)
        return window
    else:
        raise TypeError("Invalid window.")

def mavg(x, order):
    return sp.fftconvolve(x, np.full(order, 1.0 / order))[order // 2:order // 2 + len(x)]

def roundUpToPowerOf2(v):
    return int(2 ** np.ceil(np.log2(v)))

def parabolicInterpolation(input, i, val = True, overAdjust = False):
    lin = len(input)

    ret = 0.0
    if(i > 0 and i < lin - 1):
        s0 = float(input[i - 1])
        s1 = float(input[i])
        s2 = float(input[i + 1])
        a = (s0 + s2) / 2.0 - s1
        if(a == 0):
            return (i, input[i])
        b = s2 - s1 - a
        adjustment = -(b / a * 0.5)
        if(not overAdjust and abs(adjustment) > 1.0):
            adjustment = 0.0
        x = i + adjustment
        if(val):
            y = a * adjustment * adjustment + b * adjustment + s1
            return (x, y)
        else:
            return x
    else:
        x = i
        if(val):
            y = input[x]
            return (x, y)
        else:
            return x

def fixIntoUnit(x):
    if(isinstance(x, complex)):
        return (1 + 0j) / np.conj(x) if np.abs(x) > 1.0 else x
    else:
        need = np.abs(x) > 1.0
        x[need] = (1 + 0j) / np.conj(x[need])
        return x

def lerp(a, b, ratio):
    return a + (b - a) * ratio

def formantFreq(n, L = 0.168, c = 340.29):
    return (2 * n - 1) * c / 4 / L

def formantNumber(freq, L = 0.168, c = 340.29):
    return int(round((freq * 4 * L / c + 1) / 2))

def freqToMel(x, a = 2595.0, b = 700.0):
    return a * np.log10(1.0 + x / b)

def melToFreq(x, a = 2595.0, b = 700.0):
    return (np.power(10, x / a) - 1.0) * b

def freqToSemitone(freq):
    return np.log2(freq / 440.0) * 12.0 + 69.0

def semitoneToFreq(semi):
    return np.power(2, (semi - 69.0) / 12.0) * 440.0

def calcSRER(x, y):
    return np.log10(np.std(x) / np.std(x - y)) * 20.0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
