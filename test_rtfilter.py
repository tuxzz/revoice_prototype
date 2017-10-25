import numpy as np
import pylab as pl
from revoice import *
from revoice.common import *

for kernelSize in (3, 5, 7, 9, 111, 199, 255):
    kernel = np.random.uniform(0.0, 1.0, kernelSize)
    rtfilterProc = rtfilter.Procressor(kernel)
    for dataSize in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 128, 777, 999, 1024, 4096, 4097):
        data = np.random.uniform(0.0, 1.0, dataSize)
        for m in range(1, 10):
            rtOut = np.zeros(0)
            i = 0
            delayed = 0
            while(i < dataSize - dataSize % m):
                out = rtfilterProc(data[i:i + m])
                if(out is not None):
                    rtOut = np.concatenate((rtOut, out))
                else:
                    delayed += 1
                del out
                i += m
            if(m == 1 and delayed != rtfilterProc.delay and dataSize >= rtfilterProc.delay):
                print("Delay mismatch expected %d, real %d @ kernelSize = %d, dataSize = %d, m = %d" % (rtfilterProc.delay, delayed, kernelSize, dataSize, m))
                #exit()
            if(i < dataSize):
                out = rtfilterProc(data[i:])
                if(out is not None):
                    rtOut = np.concatenate((rtOut, out))
            delayed = rtfilterProc(None)
            rtOut = np.concatenate((rtOut, delayed))
            out = np.convolve(data, kernel)[kernelSize // 2:-(kernelSize // 2)]

            if(rtOut.shape != out.shape):
                print("Shape mismatch rtOut = %s, out = %s @ kernelSize = %d, dataSize = %d, m = %d" % (str(rtOut.shape), str(out.shape), kernelSize, dataSize, m))
                exit()
            if((np.abs(rtOut - out) < 1e-10).all()):
                print("Test passed @ kernelSize = %d, dataSize = %d, m = %d" % (kernelSize, dataSize, m))
            else:
                print("Wrong value @ kernelSize = %d, dataSize = %d, m = %d" % (kernelSize, dataSize, m))
                pl.plot(rtOut - out)
                pl.show()
                exit(1)
print("Everything passed")