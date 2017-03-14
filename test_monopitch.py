import numpy as np
import pylab as pl
from revoice import *
from revoice.common import *

#w, sr = loadWav("voices/renri_i_A3.wav")
w, sr = loadWav("voices/chihaya_01.wav")

pyinProc = pyin.Processor(sr, prefilter = False)
obsProbList = pyinProc(w)
monopitchProc = monopitch.Processor(*monopitch.parameterFromPYin(pyinProc))
f0List = monopitchProc(w, obsProbList)

pyinProc = pyin.Processor(sr)
obsProbList = pyinProc(w)
monopitchProc = monopitch.Processor(*monopitch.parameterFromPYin(pyinProc))
f0List_pf = monopitchProc(w, obsProbList)

t = np.arange(len(f0List)) * pyinProc.hopSize / sr
#t = np.arange(len(f0List))
pl.plot(t, f0List, label = "Direct")
pl.plot(t, f0List_pf, label = "Prefiltered")
pl.legend()
pl.show()
