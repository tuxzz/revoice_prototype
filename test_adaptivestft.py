import numpy as np
import pylab as pl
from revoice import *
from revoice.common import *

w, sr = loadWav("voices/yuri_orig.wav")

pyinProc = pyin.Processor(sr)
obsProbList = pyinProc(w)

monopitchProc = monopitch.Processor(*monopitch.parameterFromPYin(pyinProc))
f0List = monopitchProc(w, obsProbList)

stftProc = adaptivestft.Processor(sr)
magnList, phaseList, refinedF0List = stftProc(w, f0List, refineF0 = True)

pl.imshow(magnList.T, interpolation='bicubic', aspect='auto', origin='lower')
pl.plot(f0List / sr * stftProc.fftSize, label = 'original f0')
pl.plot(refinedF0List / sr * stftProc.fftSize, label = 'refined f0')
pl.legend()
pl.show()
