import numpy as np
import scipy.interpolate as ipl
import scipy.signal as sp
from .common import *

class Processor:
    def __init__(self, sr):
        self.samprate = float(sr)

    def __call__(self, timeList, f0List, hFreqList, hAmpList, hPhaseList, sinusoidEnergyList, noiseEnvList, noiseEnergyList):
        nHop = len(f0List)
        nNewHop = len(timeList)

        newF0List = np.zeros(nNewHop)
        newHFreqList = np.full((nNewHop, hFreqList.shape[1]), 1.0)
        newHAmpList = np.zeros((nNewHop, hAmpList.shape[1]))
        newHPhaseList = np.zeros((nNewHop, hPhaseList.shape[1]))
        newSinusoidEnergyList = np.zeros(nNewHop)

        segments = splitArray(f0List)
        segBeginHop = 0
        newHopIndexList = np.arange(nNewHop)
        for iSegment, segment in enumerate(segments):
            if(segment[0] > 0.0):
                segEndHop = segBeginHop + len(segment)
                newHopIndexSlice = newHopIndexList[np.logical_and(timeList >= segBeginHop, timeList < segEndHop)]
                newHopIndexBegin, newHopIndexEnd = newHopIndexSlice[0], newHopIndexSlice[-1]
                if(len(segment) == 1):
                    iHop = segment[0]
                    newF0List[newHopIndexBegin:newHopIndexEnd] = f0List[iHop]
                    newHFreqList[newHopIndexBegin:newHopIndexEnd] = hFreqList[iHop]
                    newHAmpList[newHopIndexBegin:newHopIndexEnd] = hAmpList[iHop]
                    newHPhaseList[newHopIndexBegin:newHopIndexEnd] = hPhaseList[iHop]
                    newSinusoidEnergyList[newHopIndexBegin:newHopIndexEnd] = sinusoidEnergyList[iHop]
                else:
                    iplX = np.linspace(segBeginHop, segEndHop + 1, segEndHop - segBeginHop)
                    iplY = timeList[newHopIndexBegin:newHopIndexEnd]
                    newF0List[newHopIndexBegin:newHopIndexEnd] = ipl.interp1d(iplX, f0List[segBeginHop:segEndHop], kind = 'linear')(iplY)
                    newHFreqList[newHopIndexBegin:newHopIndexEnd] = ipl.interp1d(iplX, hFreqList[segBeginHop:segEndHop], axis = 0, kind = 'linear')(iplY)
                    newHAmpList[newHopIndexBegin:newHopIndexEnd] = ipl.interp1d(iplX, hAmpList[segBeginHop:segEndHop], axis = 0, kind = 'linear')(iplY)
                    newHPhaseList[newHopIndexBegin:newHopIndexEnd] = ipl.interp1d(iplX, np.unwrap(hPhaseList[segBeginHop:segEndHop], axis = 0), axis = 0, kind = 'linear')(iplY)
                    newSinusoidEnergyList[newHopIndexBegin:newHopIndexEnd] = ipl.interp1d(iplX, sinusoidEnergyList[segBeginHop:segEndHop], kind = 'linear')(iplY)
            segBeginHop += len(segment)

        iplX = np.arange(nHop)
        newNoiseEnvList = ipl.interp1d(iplX, noiseEnvList, axis = 0, kind = 'linear')(timeList)
        newNoiseEnergyList = ipl.interp1d(iplX, noiseEnergyList, kind = 'linear')(timeList)

        return newF0List, newHFreqList, newHAmpList, newHPhaseList, newSinusoidEnergyList, newNoiseEnvList, newNoiseEnergyList
