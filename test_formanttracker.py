import numpy as np
import pylab as pl
from revoice import *
from revoice.common import *
import scipy.fftpack as sfft

w, sr = loadWav("voices/renri/i.wav")
newSr = 16000.0
resampleRatio = newSr / sr
order = 13

print("F0 Analyzing...")
pyinProc = pyin.Processor(sr)
obsProbList = pyinProc(w)

monopitchProc = monopitch.Processor(*monopitch.parameterFromPYin(pyinProc))
f0List = monopitchProc(w, obsProbList)

print("Envelope Analyzing...")
envProc = mfienvelope.Processor(sr)
envList = envProc(w, f0List)

print("LPC Analyzing...")
lpcProc = lpc.Burg(sr, resampleRatio = resampleRatio)
coeff, xms = lpcProc(w, f0List, order)

FList, bwList = lpc.toFormant(coeff, newSr)

print("Formant Tracking...")
ftProc = formanttracker.Processor(*formanttracker.parameterGenerate(5, envProc.hopSize, sr))
FList = ftProc(FList, envList, sr)

pl.imshow(envList.T, interpolation = 'bicubic', aspect = 'auto', origin = 'lower')
print(FList[100])
pl.plot(FList / sr * envProc.fftSize)
pl.show()
