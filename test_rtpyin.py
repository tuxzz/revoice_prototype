import numpy as np
import pylab as pl
from revoice import *
from revoice.common import *

#w, sr = loadWav("voices/renri_i_A3.wav")
w, sr = loadWav("voices/yuri_orig.wav")

nX = len(w)

print("Without prefilter...")
x = w
rtpyinProc = rtpyin.Processor(sr, prefilter = False)
nHop = getNFrame(nX, rtpyinProc.hopSize)
obsProbList = []
iInHop = 0
iOutHop = 0
while(True):
    data = x[iInHop * rtpyinProc.hopSize:(iInHop + 1) * rtpyinProc.hopSize]
    if(len(data) == 0):
        data = None
    out = rtpyinProc(data)
    if(out is not None):
        obsProbList.append(out)
        iOutHop += 1
    elif(data is None):
        break
    iInHop += 1
if(iOutHop != nHop):
    print("nHop mismatch(expected %d, got %d)" % (nHop, iOutHop))
    exit(1)
del x
f0List = pyin.extractF0(obsProbList)

print("With prefilter...")
x = w
rtpyinProc = rtpyin.Processor(sr, prefilter = True)
nHop = getNFrame(nX, rtpyinProc.hopSize)
obsProbList_pf = []
iInHop = 0
iOutHop = 0
while(True):
    data = x[iInHop * rtpyinProc.hopSize:(iInHop + 1) * rtpyinProc.hopSize]
    if(len(data) == 0):
        data = None
    out = rtpyinProc(data)
    if(out is not None):
        obsProbList_pf.append(out)
        iOutHop += 1
    elif(data is None):
        break
    iInHop += 1
if(iOutHop != nHop):
    print("nHop mismatch(expected %d, got %d)" % (nHop, iOutHop))
    exit(1)
del x
f0List_pf = pyin.extractF0(obsProbList_pf)

print("Non-RT without prefilter...")
pyinProc = pyin.Processor(sr, prefilter = False)
obsProbList_o = pyinProc(w, removeDC = False)
f0List_o = pyin.extractF0(obsProbList_o)

print("Non-RT with prefilter...")
pyinProc = pyin.Processor(sr, prefilter = True)
obsProbList_pf_o = pyinProc(w, removeDC = False)
f0List_pf_o = pyin.extractF0(obsProbList_pf_o)

if((np.abs(f0List - f0List_o) > 1e-5).any()):
    print("Test failed without prefilter, max diff = %lf" % np.max(np.abs(f0List - f0List_o)))
    exit(1)
for i in range(nHop):
    if((obsProbList[i] != obsProbList_o[i]).any()):
        print("First obsProb diff without prefilter @ %d(%s, %s)" % (i, obsProbList[i], obsProbList_o[i]))
        exit(1)
if((np.abs(f0List_pf - f0List_pf_o) > 1e-5).any()):
    print("Test failed with prefilter, max diff = %lf" % np.max(np.abs(f0List_pf - f0List_pf_o)))
    exit(1)
for i in range(nHop):
    if((obsProbList_pf[i] != obsProbList_pf_o[i]).any()):
        print("First obsProb diff with prefilter @ %d(%s, %s)" % (i, obsProbList[i], obsProbList_o[i]))
        exit(1)

t = np.arange(len(f0List)) * rtpyinProc.hopSize / sr
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