import numpy as np
import numba as nb

from .common import *

def difference(x):
    frameSize = len(x)
    outSize = frameSize // 2
    out = np.zeros(outSize, dtype = np.float64)

    # POWER TERM CALCULATION
    # ... for the power terms in equation (7) in the Yin paper

    powerTerms = np.zeros(outSize, dtype = np.float64)
    powerTerms[0] = np.sum(x[:outSize] ** 2)

    for i in range(1, outSize):
        powerTerms[i] = powerTerms[i - 1] - x[i - 1] * x[i - 1] + x[i + outSize] * x [i + outSize]

    # YIN-STYLE ACF via FFT
    # 1. data
    transformedAudio = np.fft.rfft(x)

    # 2. half of the data, disguised as a convolution kernel
    kernel = np.zeros((frameSize), dtype = np.float64)
    kernel[:outSize] = x[:outSize][::-1]
    transformedKernel = np.fft.rfft(kernel)

    # 3. convolution
    yinStyleACF = transformedAudio * transformedKernel
    transformedAudio = np.fft.irfft(yinStyleACF)

    # CALCULATION OF difference function
    # according to (7) in the Yin paper
    out = powerTerms[0] + powerTerms - 2 * transformedAudio[outSize - 1:-1]
    return out

@nb.jit(nb.float64[:](nb.float64[:]), cache=True)
def cumulativeDifference(x):
    out = x.copy()
    nOut = len(out)

    out[0] = 1.0
    sum = 0.0

    for i in range(1, nOut):
        sum += out[i]
        if(sum == 0.0):
            out[i] = 1
        else:
            out[i] *= i / sum
    return out

def findValleys(x, minFreq, maxFreq, sr, threshold = 0.5, step = 0.01):
    ret = []
    begin = max(1, int(sr / maxFreq))
    end = min(len(x) - 1, int(np.ceil(sr / minFreq)))
    for i in range(begin, end):
        prev = x[i - 1]
        curr = x[i]
        next = x[i + 1]
        if(prev > curr and next > curr and curr < threshold):
            threshold = curr - step
            ret.append(i)
    return ret

class Processor:
    def __init__(self, sr):
        self.samprate = float(sr)
        self.hopSize = roundUpToPowerOf2(self.samprate * 0.0025)
        self.windowSize = roundUpToPowerOf2(self.samprate * 0.025)

        self.minFreq = 80.0
        self.maxFreq = 1000.0

        self.valleyThreshold = 0.5
        self.valleyStep = 0.01

    def processSingle(self, x):
        nX = len(x)
        buffSize = nX // 2
        sr = self.samprate

        buff = difference(x)
        buff = cumulativeDifference(buff)

        valleys = findValleys(buff, self.minFreq, self.maxFreq, sr, threshold = self.valleyThreshold, step = self.valleyStep)

        if(valleys):
            return sr / parabolicInterpolation(buff, valleys[-1], val = False)
        else:
            return 0.0

    def __call__(self, x, removeDC = True):
        nX = len(x)
        nHop = getNFrame(nX, self.hopSize)

        out = np.zeros(nHop, dtype = np.float64)
        for iHop in range(nHop):
            frame = getFrame(x, iHop * self.hopSize, self.windowSize)
            if(removeDC):
                frame = simpleDCRemove(frame)
            out[iHop] = self.processSingle(frame)

        return out
