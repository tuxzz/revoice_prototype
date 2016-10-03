import numpy as np
import pylab as pl
from revoice import *
from revoice.common import *
import scipy.fftpack as sfft

w, sr = loadWav("voices/yuri_orig.wav")

pyinProc = pyin.Processor(sr)
obsProbList = pyinProc(w)

monopitchProc = monopitch.Processor(*monopitch.parameterFromPYin(pyinProc))
f0List = monopitchProc(w, obsProbList)

envProc = melenvelope.Processor(sr)
env = envProc(w, f0List)

pl.imshow(env.T, interpolation='bicubic', aspect='auto', origin='lower')
pl.plot(f0List / sr * envProc.fftSize)
pl.figure()
pl.plot(env[100])
pl.show()
