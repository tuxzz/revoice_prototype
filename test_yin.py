import numpy as np
import pylab as pl
from revoice import *
from revoice.common import *

#w, sr = loadWav("voices/renri_i_A3.wav")
w, sr = loadWav("voices/yuri_orig.wav")

yinProc = yin.Processor(sr, prefilter = False)
f0List = yinProc(w)

yinProc = yin.Processor(sr)
f0List_pf = yinProc(w)

t = np.arange(len(f0List)) * yinProc.hopSize / sr
#t = np.arange(len(f0List))
pl.plot(t, f0List, label = "Direct")
pl.plot(t, f0List_pf, label = "Prefiltered")
pl.legend()
pl.show()
