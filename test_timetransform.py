import numpy as np
import pylab as pl
from revoice import *
from revoice.common import *

w, sr = loadWav("voices/renri_i_A3.wav")
#w, sr = loadWav("voices/chihaya_01.wav")

print("F0 Analyzing...")
pyinProc = pyin.Processor(sr)
obsProbList = pyinProc(w)
monopitchProc = monopitch.Processor(*monopitch.parameterFromPYin(pyinProc))
f0List = monopitchProc(w, obsProbList)

print("HNM Analyzing...")
hnmProc = hnm.Analyzer(sr, harmonicAnalysisMethod = "qhmair")
f0List, hFreqList, hAmpList, hPhaseList, sinusoidEnergyList, noiseEnvList, noiseEnergyList = hnmProc(w, f0List)

print("Time Transforming...")
timeProc = timetransform.Processor(sr)
newTimeList = np.linspace(0, len(f0List) - 1, len(f0List) * 4.0)
f0List, hPhaseList, hFreqList, hAmpList, sinusoidEnergyList, noiseEnvList, noiseEnergyList = timeProc.simple(newTimeList, f0List, hPhaseList, ((hFreqList, 1.0), hAmpList, sinusoidEnergyList, noiseEnvList, noiseEnergyList))

print("Synthing...")
synProc = hnm.Synther(sr)
synthed = synProc(f0List, hFreqList, hAmpList, hPhaseList, sinusoidEnergyList, noiseEnvList, noiseEnergyList)

harmonicList = hFreqList * f0List.reshape(len(f0List), 1) * np.arange(1, hFreqList.shape[1] + 1)
harmonicList[np.logical_or(harmonicList <= 0.0, harmonicList > 20e3)] = np.nan
pl.figure()
pl.plot(harmonicList)
pl.figure()
pl.plot(sinusoidEnergyList)
pl.plot(noiseEnergyList)
pl.figure()
pl.plot(synthed)
pl.show()
