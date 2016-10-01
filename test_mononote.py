import numpy as np
import pylab as pl
from revoice import *
from revoice.common import *
import matplotlib.patches as patches

#w, sr = loadWav("voices/renri_i_A3.wav")
#w, sr = loadWav("voices/yuri_orig.wav")
w, sr = loadWav("voices/tuku_ra.wav")

pyinProc = pyin.Processor(sr)
obsProbList = pyinProc(w)

monopitchProc = monopitch.Processor(*monopitch.parameterFromPYin(pyinProc))
f0 = monopitchProc(w, obsProbList)

mononoteProc = mononote.Processor(*mononote.parameterFromPYin(pyinProc))
noteList = mononoteProc(w, obsProbList)

fig = pl.figure()
ax = pl.subplot(111)
pl.plot(f0)
for iNote, note in enumerate(noteList):
    freq = semitoneToFreq(note["pitch"])
    begin = note["begin"]
    end = note["end"]
    ax.add_patch(patches.Rectangle((begin, freq - 5.0), end - begin, 10.0, alpha=0.39))
pl.show()
