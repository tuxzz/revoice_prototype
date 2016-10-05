import numpy as np
import pylab as pl
from revoice import *
from revoice.common import *

#w, sr = loadWav("voices/renri_i_A3.wav")
w, sr = loadWav("voices/yuri_orig.wav")

pyinProc = pyin.Processor(sr)
obsProbList = pyinProc(w)
f0List = pyinProc.extractF0(obsProbList)

pl.plot(f0List)
pl.show()
