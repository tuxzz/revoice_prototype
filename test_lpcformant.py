import numpy as np
import pylab as pl
from revoice import *
from revoice.common import *
import scipy.fftpack as sfft

w, sr = loadWav("voices/renri/i.wav")
newSr = 16000.0
resampleRatio = newSr / sr
order = 13
fftSize = 1024

print("F0 Analyzing...")
pyinProc = pyin.Processor(sr)
obsProbList = pyinProc(w)

monopitchProc = monopitch.Processor(*monopitch.parameterFromPYin(pyinProc))
f0List = monopitchProc(w, obsProbList)

print("LPC Analyzing...")
lpcProc = lpc.Burg(sr, resampleRatio = resampleRatio)
coeff, xms = lpcProc(w, f0List, order)

lpcSpectrum = lpc.toSpectrum(coeff, xms, lpcProc.preEmphasisFreq, fftSize, newSr)
FList, bwList = lpc.toFormant(coeff, newSr)

lpcSpectrum = np.log(np.clip(lpcSpectrum, 1e-6, np.inf))
pl.imshow(lpcSpectrum.T, interpolation = 'bicubic', aspect = 'auto', origin = 'lower')
pl.plot(FList / newSr * fftSize, 'o')
pl.show()
