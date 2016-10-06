import numpy as np
import pylab as pl
from revoice import *
from revoice.common import *

#w, sr = loadWav("voices/renri_i_A3.wav")
w, sr = loadWav("voices/yuri_orig.wav")

print("F0 Analyzing...")
pyinProc = pyin.Processor(sr)
obsProbList = pyinProc(w)
monopitchProc = monopitch.Processor(*monopitch.parameterFromPYin(pyinProc))
f0List = monopitchProc(w, obsProbList)

print("HNM Analyzing...")
hnmProc = hnm.Analyzer(sr, harmonicAnalysisMethod = "qfft")
f0List, hFreqList, hAmpList, hPhaseList, sinusoidEnergyList, noiseEnvList, noiseEnergyList = hnmProc(w, f0List)

print("HNM Synthing...")
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
