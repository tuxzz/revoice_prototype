import numpy as np
import scipy.signal as sp

from .common import *

class Processor:
    windows = {
        'hanning': (sp.hanning, 1.5),
        'blackman': (sp.blackman, 1.73),
        'blackmanharris': (sp.blackmanharris, 2.0),
    }

    def __init__(self, sr, window = 'blackman'):
        self.samprate = float(sr)
        self.hopSize = roundUpToPowerOf2(self.samprate * 0.0025)
        self.fftSize = roundUpToPowerOf2(self.samprate * 0.05)
        self.peakSearchRange = 0.3
        if(type(window) is str):
            self.window = self.windows[window]
        elif(type(window) is tuple):
            self.window = window
            assert(len(self.window) == 2)
        else:
            raise TypeError("Invalid window.")

    def __call__(self, x, f0List, removeDC = True, refineF0 = False, bFac = 1.0):
        # constant
        nX = len(x)
        nHop = getNFrame(nX, self.hopSize)
        nBin = self.fftSize // 2 + 1
        windowFunc, B = self.window
        B *= bFac

        # check input
        assert(nHop == len(f0List))
        assert(self.fftSize % 2 == 0)

        # do calculate
        magnList = np.zeros((nHop, nBin), dtype = np.float64)
        phaseList = np.zeros((nHop, nBin), dtype = np.float64)
        if(refineF0):
            refinedF0List = np.zeros(nHop, dtype = np.float64)
        for iHop, f0 in enumerate(f0List):
            if(f0 > 0.0):
                windowSize = int(min(self.fftSize, np.ceil(self.samprate / f0) * B * 2.0))
                if(windowSize % 2 != 0):
                    windowSize += 1
                halfWindowSize = windowSize // 2
            else:
                windowSize = self.hopSize * 2
                halfWindowSize = self.hopSize
            window = windowFunc(windowSize)
            windowNormFac = 2.0 / np.sum(window)
            frame = getFrame(x, iHop * self.hopSize, windowSize)
            frame *= window
            if(removeDC):
                frame = simpleDCRemove(frame)

            tSig = np.zeros(self.fftSize, dtype = np.float64)
            tSig[:halfWindowSize] = frame[halfWindowSize:]
            tSig[-halfWindowSize:] = frame[:halfWindowSize]
            fSig = np.fft.rfft(tSig)
            magnList[iHop] = np.abs(fSig) * windowNormFac
            phaseList[iHop] = np.unwrap(np.angle(fSig))

            if(refineF0 and f0 > 0.0):
                lowerIdx = max(0, int(np.floor(f0 * self.fftSize / self.samprate * (1.0 - self.peakSearchRange))))
                upperIdx = min(self.fftSize // 2, int(np.floor(f0 * self.fftSize / self.samprate * (1.0 + self.peakSearchRange))))
                peakIdx = np.argmax(magnList[iHop][lowerIdx:upperIdx]) + lowerIdx

                frame = getFrame(x, iHop * self.hopSize - 1, windowSize)
                frame *= window
                if(removeDC):
                    frame = simpleDCRemove(frame)

                tSig = np.zeros(self.fftSize, dtype = np.float64)
                tSig[:halfWindowSize] = frame[halfWindowSize:]
                tSig[-halfWindowSize:] = frame[:halfWindowSize]
                deltaPhase = np.unwrap(np.angle(np.fft.rfft(tSig)))[peakIdx]
                phase = phaseList[iHop][peakIdx]

                phase -= np.floor(phase / 2.0 / np.pi) * 2.0 * np.pi
                deltaPhase -= np.floor(deltaPhase / 2.0 / np.pi) * 2.0 * np.pi
                if(phase < deltaPhase):
                    phase += 2 * np.pi
                refinedF0 = (phase - deltaPhase) / 2.0 / np.pi * self.samprate
                if(np.abs(refinedF0 - f0) / f0 > 0.08 or np.abs(refinedF0 / self.samprate * self.fftSize - peakIdx) > 1.0):
                    refinedF0List[iHop] = f0
                else:
                    refinedF0List[iHop] = refinedF0
        ret = [magnList, phaseList]
        if(refinedF0):
            ret.append(refinedF0List)
        return tuple(ret)
