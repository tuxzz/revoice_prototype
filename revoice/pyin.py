import numpy as np

from .common import *
from . import yin

def normalized_pdf(a, b, begin, end, number):
    x = np.arange(0, number, dtype = np.float64) * ((end - begin) / number)
    v = np.power(x, a - 1.0) * np.power(1.0 - x, b - 1.0)
    for i in range(2, len(v) + 1):
        i = len(v) - i
        if(v[i] < v[i + 1]):
            v[i] = v[i + 1]
    return v / np.sum(v)

class Processor:
    def __init__(self, sr, **kwargs):
        self.samprate = float(sr)
        self.hopSize = kwargs.get("hopSize", roundUpToPowerOf2(self.samprate * 0.0025))
        self.windowSize = kwargs.get("windowSize", roundUpToPowerOf2(self.samprate * 0.025))

        self.minFreq = kwargs.get("minFreq", 80.0)
        self.maxFreq = kwargs.get("maxFreq", 1000.0)

        self.valleyThreshold = kwargs.get("valleyThreshold", 0.5)
        self.valleyStep = kwargs.get("valleyStep", 0.01)

        self.probThreshold = kwargs.get("probThreshold", 0.02)
        self.weightPrior = kwargs.get("weightPrior", 5.0)
        self.bias = kwargs.get("bias", 1.0)

        self.pdf = kwargs.get("pdf", normalized_pdf(1.7, 6.8, 0.0, 1.0, 128))

    def processSingle(self, x):
        nX = len(x)
        buffSize = nX // 2
        sr = self.samprate
        pdfSize = len(self.pdf)

        buff = yin.difference(x)
        buff = yin.cumulativeDifference(buff)

        valleys = yin.findValleys(buff, self.minFreq, self.maxFreq, sr, threshold = self.valleyThreshold, step = self.valleyStep)
        nValley = len(valleys)

        freqProb = np.zeros((nValley, 2), dtype = np.float64)
        probTotal = 0.0
        weightedProbTotal = 0.0
        for iValley, valley in enumerate(valleys):
            ipledIdx, ipledVal = parabolicInterpolation(buff, valley)
            freq = self.samprate / ipledIdx
            v0 = 1 if(iValley == 0) else min(1.0, buff[valleys[iValley - 1]] + 1e-10)
            v1 = 0 if(iValley == nValley - 1) else max(0.0, buff[valleys[iValley + 1]]) + 1e-10
            prob = 0.0
            for i in range(int(v1 * pdfSize), int(v0 * pdfSize)):
                prob += self.pdf[i] * (1.0 if(ipledVal < i / pdfSize) else 0.01)
            prob = min(prob, 0.99)
            prob *= self.bias
            probTotal += prob
            if(ipledVal < self.probThreshold):
                prob *= self.weightPrior
            weightedProbTotal += prob
            freqProb[iValley] = freq, prob

        # renormalize
        if(nValley > 0 and weightedProbTotal != 0.0):
            freqProb.T[1] *= probTotal / weightedProbTotal
        return freqProb

    def __call__(self, x, removeDC = True):
        nX = len(x)
        nHop = getNFrame(nX, self.hopSize)

        out = []
        for iHop in range(nHop):
            frame = getFrame(x, iHop * self.hopSize, self.windowSize)
            if(removeDC):
                frame = simpleDCRemove(frame)
            out.append(self.processSingle(frame))

        return out
