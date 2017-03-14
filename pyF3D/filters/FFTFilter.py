import numpy as np
import pkg_resources as pkg
import pyopencl as cl
import time
import pyF3D.FilterClasses as fc

class FFTFilter:

    FFTChoice = ['Forward', 'Inverse']

    def __init__(self, FFTChoice='Forward'):

        if FFTChoice not in self.FFTChoice:
            raise ValueError("'FFTChoice' parameter must be either 'Forward' or 'Inverse'")

        self.name = 'FFTFilter'
        self.selectedFFTChoice = FFTChoice

        # load clattr from RunnablePOCLFilter
        self.clattr = None
        self.atts = None #load attributes from RunnablePOCLFilter

    def toJSONString(self):
        result = "{ + \"Name\" : \"" + self.getName() + "\" , "
        result += "\"fftChoice\" : \"" + str(self.selectedFFTChoice) +"\" , }"
        return result

    def clone(self):
        return FFTFilter(FFTChoice=self.selectedFFTChoice)

    def getInfo(self):
        info = fc.FilterInfo()
        info.name = self.getName()
        info.memtype = float
        info.useTempBuffer = True
        info.overlapX = info.overlapY = info.overlapZ = 0
        return info

    def getName(self):
        return 'FFTFilter'

    def loadKernel(self):
        try:
            filename = "OpenCL/FFTFilter.cl"
            program = cl.Program(self.clattr.context, pkg.resource_string(__name__, filename)).build()
        except Exception as e:
            raise e

        self.kernel = cl.Kernel(program, "FFTFilter")
        return True

    def runFilter(self):

        direction = 1 if self.selectedFFTChoice == 'Forward' else -1

        globalSize = [0, 0]
        localSize = [0, 0]
        self.clattr.computeWorkingGroupSize(localSize, globalSize, [self.atts.width, self.atts.height, 1])

        try:
            self.kernel.set_args(self.clattr.inputBuffer, self.clattr.outputBuffer, self.clattr.outputTmpBuffer,
                                 np.int32(direction), np.int32(self.atts.width), np.int32(self.atts.height),
                                 np.int32(self.clattr.maxSliceCount), np.int32(2))
            cl.enqueue_nd_range_kernel(self.clattr.queue, self.kernel, globalSize, localSize)
        except Exception as e:
            raise e

        # write results
        cl.enqueue_copy(self.clattr.queue, self.clattr.inputBuffer, self.clattr.outputBuffer)
        self.clattr.queue.finish()
        return True



    def setAttributes(self, CLAttributes, atts, index):
        self.clattr = CLAttributes
        self.atts = atts
        self.index = index



