import numpy as np
import pylab as pl
from revoice import *
from revoice.common import *

t = np.linspace(0, 0.008, 16384)
f = np.linspace(10.0, 4000.0, 16384 * 16)
T0 = 0.008
sr = 8e3

pl.figure()
pl.subplot(111)
pl.plot(t, lfmodel.calcFlowDerivative(t, T0, 1.0, *lfmodel.fromRd(0.3)), label = "Rd = 0.3")
pl.plot(t, lfmodel.calcFlowDerivative(t, T0, 1.0, *lfmodel.fromRd(1.0)), label = "Rd = 1.0")
pl.plot(t, lfmodel.calcFlowDerivative(t, T0, 1.0, *lfmodel.fromRd(2.5)), label = "Rd = 2.5")
pl.legend()
pl.figure()
pl.plot(f, np.log(lfmodel.calcSpectrum(f, sr, T0, 1.0, *lfmodel.fromRd(0.3))[0]), label = "Rd = 0.3")
pl.plot(f, np.log(lfmodel.calcSpectrum(f, sr, T0, 1.0, *lfmodel.fromRd(1.0))[0]), label = "Rd = 1.0")
pl.plot(f, np.log(lfmodel.calcSpectrum(f, sr, T0, 1.0, *lfmodel.fromRd(2.5))[0]), label = "Rd = 2.5")
pl.legend()
pl.show()
