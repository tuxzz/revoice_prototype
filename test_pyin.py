import numpy as np
import pylab as pl
from revoice import *
from revoice.common import *

#w, sr = loadWav("voices/renri_i_A3.wav")
w, sr = loadWav("voices/yuri_orig.wav")

pyinProc = pyin.Processor(sr, prefilter = False)
obsProbList = pyinProc(w)
f0List = pyin.extractF0(obsProbList)

pyinProc = pyin.Processor(sr)
obsProbList = pyinProc(w)
f0List_pf = pyin.extractF0(obsProbList)

t = np.arange(len(f0List)) * pyinProc.hopSize / sr
#t = np.arange(len(f0List))
pl.plot(t, f0List, label = "Direct")
pl.plot(t, f0List_pf, label = "Prefiltered")
pl.legend()
pl.show()
