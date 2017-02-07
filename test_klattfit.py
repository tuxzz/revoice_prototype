import numpy as np
import pylab as pl
from revoice import *
from revoice.common import *

w, sr = loadWav("voices/tuku/e.wav")
maxFreq = 8000.0
iHop = 150

print("F0 analyzing...")
pyinProc = pyin.Processor(sr)
obsProbList = pyinProc(w)

monopitchProc = monopitch.Processor(*monopitch.parameterFromPYin(pyinProc))
f0List = monopitchProc(w, obsProbList)

print("HNM Analyzing...")
hnmProc = hnm.Analyzer(sr, harmonicAnalysisMethod = "qfft")
f0List, hFreqList, hAmpList, hPhaseList, sinusoidEnergyList, noiseEnvList, noiseEnergyList = hnmProc(w, f0List)
harmonicList = np.arange(1, hFreqList.shape[1] + 1) * f0List[iHop]

print("Envelope analyzing...")
stftProc = adaptivestft.Processor(sr)
magnList, _, f0List = stftProc(w, f0List, refineF0 = True)

envProc = mfienvelope.Processor(sr)
envList = envProc(w, f0List)

print("Klatt fitting...")
maxHar = np.arange(0, hFreqList.shape[1])[harmonicList < maxFreq][-1]
maxBin = int(round(maxFreq / sr * envProc.fftSize))

xSpectrum = (np.arange(envList.shape[1]) / envProc.fftSize * sr)[:maxBin]
xHarmonic = harmonicList[:maxHar]

yEnvelope = np.exp(envList[iHop][:maxBin]) * preEmphasisResponse(xSpectrum, 50.0, sr)
ySpectrum = magnList[iHop][:maxBin] * preEmphasisResponse(xSpectrum, 50.0, sr)
yHarmonic = hAmpList[iHop][:maxHar] * preEmphasisResponse(xHarmonic, 50.0, sr)

procPO = klatt.ParameterOptimizer(sr)
porg = klatt.ParameterOptimizer.defaultPreOptimizeReferenceGetter(yEnvelope, envProc.fftSize, sr)

FList = formantFreq(np.arange(1, 6), L = 0.16)
bwList = np.full(5, 300.0)
ampList = porg(FList)

FList, bwList, ampList = procPO.optimizeAutoJitter(xHarmonic, yHarmonic, FList, bwList, preOptimizeReferenceGetter = porg)
print(FList)
print(bwList)
print(ampList)

pl.plot(xSpectrum, np.log(np.clip(ySpectrum, 1e-6, np.inf)))
pl.plot(xSpectrum, np.log(np.clip(yEnvelope, 1e-6, np.inf)))
pl.plot(xSpectrum, np.log(np.clip(klatt.spectrumFromFilterList(xSpectrum, FList, bwList, ampList, sr), 1e-6, np.inf)))
pl.show()
