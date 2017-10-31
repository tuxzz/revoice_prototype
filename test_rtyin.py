import numpy as np
import pylab as pl
from revoice import *
from revoice.common import *

#w, sr = loadWav("voices/renri_i_A3.wav")
w, sr = loadWav("voices/yuri_orig.wav")

nX = len(w)

print("Without prefilter...")
rtyinProc = rtyin.Processor(sr, prefilter = False)
nHop = getNFrame(nX, rtyinProc.hopSize)
f0List = np.zeros(nHop)
iInHop = 0
iOutHop = 0
while(True):
    data = w[iInHop * rtyinProc.hopSize:(iInHop + 1) * rtyinProc.hopSize]
    if(len(data) == 0):
        data = None
    out = rtyinProc(data)
    if(out is not None):
        f0List[iOutHop] = out
        iOutHop += 1
    elif(data is None):
        break
    iInHop += 1
if(iOutHop != nHop):
    print("nHop mismatch(expected %d, got %d)" % (nHop, iOutHop))
    exit(1)

print("With prefilter...")
rtyinProc = rtyin.Processor(sr, prefilter = True)
nHop = getNFrame(nX, rtyinProc.hopSize)
f0List_pf = np.zeros(nHop)
iInHop = 0
iOutHop = 0
while(True):
    data = w[iInHop * rtyinProc.hopSize:(iInHop + 1) * rtyinProc.hopSize]
    if(len(data) == 0):
        data = None
    out = rtyinProc(data)
    if(out is not None):
        f0List_pf[iOutHop] = out
        iOutHop += 1
    elif(data is None):
        break
    iInHop += 1
if(iOutHop != nHop):
    print("nHop mismatch(expected %d, got %d)" % (nHop, iOutHop))
    exit(1)

print("Non-RT...")
yinProc = yin.Processor(sr, prefilter = False)
f0List_o = yinProc(w, removeDC = False)

yinProc = yin.Processor(sr, prefilter = True)
f0List_pf_o = yinProc(w, removeDC = False)

if((np.abs(f0List - f0List_o) > 1e-5).any()):
    print("Test failed without prefilter, max diff = %lf" % np.max(np.abs(f0List - f0List_o)))
if((np.abs(f0List_pf - f0List_pf_o) > 1e-5).any()):
    print("Test failed with prefilter, max diff = %lf" % np.max(np.abs(f0List_pf - f0List_pf_o)))

t = np.arange(len(f0List)) * rtyinProc.hopSize / sr
#t = np.arange(len(f0List))
pl.figure()
pl.plot(t, f0List, label = "Direct")
pl.plot(t, f0List_o, label = "Direct non-rt")
pl.legend()
pl.figure()
pl.plot(t, f0List_pf, label = "Prefiltered")
pl.plot(t, f0List_pf_o, label = "Prefiltered non-rt")
pl.legend()
pl.show()