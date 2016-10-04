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

stftProc = adaptivestft.Processor(sr)
magnList, _ = stftProc(w, f0List)

envProc = melenvelope.Processor(sr)
env = envProc(w, f0List)

pl.imshow(env.T, interpolation = 'bicubic', aspect = 'auto', origin = 'lower')
pl.plot(f0List / sr * envProc.fftSize)
pl.figure()
pl.plot(np.log(np.clip(magnList[100], 1e-6, np.inf)))
pl.plot(env[100])
pl.show()
