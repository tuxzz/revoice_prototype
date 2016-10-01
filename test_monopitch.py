import numpy as np
import pylab as pl
from revoice import *
from revoice.common import *

#w, sr = loadWav("voices/renri_i_A3.wav")
w, sr = loadWav("voices/yuri_orig.wav")

pyinProc = pyin.Processor(sr)
obsProb = pyinProc(w)

monopitchProc = monopitch.Processor(*monopitch.parameterFromPYin(pyinProc))
f0 = monopitchProc(w, obsProb)

pl.plot(f0)
pl.show()