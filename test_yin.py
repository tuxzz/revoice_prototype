import numpy as np
import pylab as pl
from revoice import *
from revoice.common import *

#w, sr = loadWav("voices/renri_i_A3.wav")
w, sr = loadWav("voices/yuri_orig.wav")

yinProc = yin.Processor(sr)
f0 = yinProc(w)

pl.plot(f0)
pl.show()
