import numpy as np
import scipy.stats as ss

from .common import *
from . import sparsehmm

# * case state % 3:
#    0: attack
#    1: stable
#    2: silent
# * possible transitions
#    attack -> attack
#    attack -> stable
#    stable -> stable
#    stable -> silent
#    silent -> silent
#    silent -> attack

def parameterFromPYin(pyin):
    hopSize = pyin.hopSize
    windowSize = pyin.windowSize
    minSemitone = int(freqToSemitone(pyin.minFreq))
    nSemitone = int(np.ceil(freqToSemitone(pyin.maxFreq)) - minSemitone)
    return hopSize, windowSize, minSemitone, nSemitone

class Processor:
    def __init__(self, hopSize, windowSize, minSemitone, nSemitone, **kwargs):
        self.hopSize = int(hopSize)
        self.windowSize = int(windowSize)
        self.minSemitone = minSemitone
        self.nSemitone = nSemitone

        self.binPerSemitone = kwargs.get("binPerSemitone", 5)
        self.probAttackTransSelf = kwargs.get("probAttackTransSelf", 0.9)
        self.probStableTransSelf = kwargs.get("probStableTransSelf", 0.99)
        self.probSilentTransSelf = kwargs.get("probSilentTransSelf", 0.9999)
        self.probStableToSilent = kwargs.get("probStableToSilent", 0.1)
        self.noteSigma = kwargs.get("noteSigma", 0.7)
        self.priorPitchedProb = kwargs.get("priorPitchedProb", 0.7)
        self.priorWeight = kwargs.get("priorWeight", 0.5)
        self.minTransSemitone = kwargs.get("minTransSemitone", 0.5)
        self.maxTransSemitone = kwargs.get("maxTransSemitone", 13.0)
        self.yinAttackSigma = kwargs.get("yinAttackSigma", 5.0)
        self.yinStableSigma = kwargs.get("yinStableSigma", 0.8)
        self.yinTrust = kwargs.get("yinTrust", 0.1)

        self.viterbiDecoder = self.createModel()

    def createModel(self):
        nBin = int(self.nSemitone * self.binPerSemitone)
        minTransBin = int(round(self.minTransSemitone * self.binPerSemitone))
        maxTransBin = int(round(self.maxTransSemitone * self.binPerSemitone))

        nState = nBin * 3
        nTrans = nBin * 5
        for iBin in range(nBin):
            begin = max(0, iBin - maxTransBin + 1)
            end = min(nBin - 1, iBin + maxTransBin)
            for jBin in range(begin, end):
                distance = abs(iBin - jBin)
                if(distance > minTransBin or distance == 0):
                    nTrans += 1

        init = np.zeros(nState, dtype = np.float64)
        frm = np.zeros(nTrans, dtype = np.int)
        to = np.zeros(nTrans, dtype = np.int)
        transProb = np.zeros(nTrans, dtype = np.float64)

        init[np.arange(2, nState, step = 3)] = 1.0 / nBin # only start from silent state

        iA = 0
        for iBin in range(nBin):
            idx = iBin * 3

            # trans to self
            frm[iA] = idx
            to[iA] = idx
            transProb[iA] = self.probAttackTransSelf

            frm[iA + 1] = idx
            to[iA + 1] = idx + 1
            transProb[iA + 1] = 1.0 - self.probAttackTransSelf

            frm[iA + 2] = idx + 1
            to[iA + 2] = idx + 1
            transProb[iA + 2] = self.probStableTransSelf

            frm[iA + 3] = idx + 1
            to[iA + 3] = idx + 2
            transProb[iA + 3] = self.probStableToSilent

            frm[iA + 4] = idx + 2
            to[iA + 4] = idx + 2
            transProb[iA + 4] = self.probSilentTransSelf

            iA += 5

            # silent to attack
            beginIA = iA
            silentProbSum = 0.0
            begin = max(0, iBin - maxTransBin + 1)
            end = min(nBin - 1, iBin + maxTransBin)
            for jBin in range(begin, end):
                distance = abs(iBin - jBin)
                if(distance > minTransBin or distance == 0):
                    prob = ss.norm.pdf(distance / self.binPerSemitone, loc = 0.0, scale = self.noteSigma)
                    silentProbSum += prob
                    frm[iA] = idx + 2
                    to[iA] = jBin * 3
                    transProb[iA] = (1.0 - self.probSilentTransSelf) * prob
                    iA += 1
            transProb[beginIA:iA] /= silentProbSum

        return sparsehmm.ViterbiDecoder(init, frm, to, transProb)

    def calcStateProb(self, obsProb, meanEnergy):
        obsProb = np.asarray(obsProb)
        nState = len(self.viterbiDecoder.init)
        assert(obsProb.ndim == 2)
        assert(obsProb.shape[1] == 2)

        nBin = int(self.nSemitone * self.binPerSemitone)
        out = np.zeros(nState, dtype = np.float64)

        probPitched = np.sum(obsProb.T[1]) * (1.0 - self.priorWeight) + self.priorPitchedProb * self.priorWeight
        pitches = freqToSemitone(obsProb.T[0])

        if(len(obsProb) > 0):
            probSum = 0.0
            for iState in range(nState):
                if(iState % 3 != 2): # if not silent
                    pitch = self.minSemitone + (iState // 3) / self.binPerSemitone
                    dists = np.abs(pitches - pitch)
                    iBest = np.argmin(dists)
                    sigma = self.yinAttackSigma if(iState % 3 == 0) else self.yinStableSigma
                    prob = np.power(obsProb[iBest][1], self.yinTrust) * ss.norm.pdf(pitches[iBest], loc = pitch, scale = sigma)
                    probSum += prob
                    out[iState] = prob
        else:
            probSum = 2.0 * nBin
            out[np.arange(0, nState, step = 3)] = 1.0
            out[np.arange(1, nState, step = 3)] = 1.0
        if(probSum > 0.0):
            out[np.arange(0, nState, step = 3)] *= probPitched / probSum
            out[np.arange(1, nState, step = 3)] *= probPitched / probSum
        out[np.arange(0, nState, step = 3)] += meanEnergy
        out[np.arange(1, nState, step = 3)] += meanEnergy
        out[np.arange(2, nState, step = 3)] = (1.0 - probPitched) / nBin

        return out

    def __call__(self, x, obsProbList):
        # constant
        nX = len(x)
        nHop = getNFrame(nX, self.hopSize)
        nBin = int(self.nSemitone * self.binPerSemitone)
        nState = len(self.viterbiDecoder.init)

        # check input
        assert(nHop == len(obsProbList))

        # decode
        obsSeq = np.zeros((nHop, nState), dtype = np.float64)
        for iHop in range(nHop):
            frame = getFrame(x, iHop * self.hopSize, self.windowSize)
            frame = simpleDCRemove(frame)
            meanEnergy = np.mean(frame ** 2)
            obsSeq[iHop] = self.calcStateProb(obsProbList[iHop], meanEnergy)
        path = self.viterbiDecoder(obsSeq)
        del obsSeq

        # track note
        noteList = []
        currNote = None
        lastState = 2
        for iHop in range(nHop):
            state = path[iHop] % 3
            if(state == 0): # attack
                currNote = {}
                currNote["begin"] = iHop
                currNote["pitch"] = (path[iHop] // 3) / self.binPerSemitone + self.minSemitone
            elif(state == 2 and lastState == 1):
                currNote["end"] = iHop + 1
                noteList.append(currNote)
                currNote = None
            lastState = state
        if(not currNote is None):
            currNote["end"] = nHop - 1
            noteList.append(currNote)
            currNote = None

        return noteList
