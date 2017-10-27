import numpy as np
import pylab as pl
from revoice import *
from revoice.common import *

#w, sr = loadWav("voices/renri_i_A3.wav")
w, sr = loadWav("voices/chihaya_01.wav")

print("pYIN...")
pyinProc = pyin.Processor(sr)
obsProbList = pyinProc(w)

print("RT...")
x = w
nX = len(x)
nHop = getNFrame(nX, pyinProc.hopSize)

rtmonopitchProc = rtmonopitch.Processor(*monopitch.parameterFromPYin(pyinProc))
f0List = np.zeros(nHop)
for iHop in range(nHop):
    frame = getFrame(x, iHop * rtmonopitchProc.hopSize, 2 * rtmonopitchProc.hopSize)
    out = rtmonopitchProc(frame, obsProbList[iHop])
    if(out is None):
        continue

    nOut = len(out)
    f0List[iHop - len(out) + 1:iHop + 1] = out

print("Non-RT...")
monopitchProc = monopitch.Processor(*monopitch.parameterFromPYin(pyinProc))
f0List_o = monopitchProc(w, obsProbList)

zF0List = f0List.copy()
zF0List[f0List < 0.0] = 0.0

zF0List_o = f0List_o.copy()
zF0List_o[f0List_o < 0.0] = 0.0

ret = 0
if((zF0List != zF0List_o).any()):
    print("Test failed.")
    ret = 1
else:
    print("Test passed")

t = np.arange(len(f0List)) * pyinProc.hopSize / sr
#t = np.arange(len(f0List))
pl.plot(t, f0List, label = "rt")
pl.plot(t, f0List_o, label = "non-rt")
pl.legend()
pl.show()
exit(ret)