import numpy as np
import pkg_resources as pkg
import pyopencl as cl
import time
from pyF3D import helpers

class MaskFilter:

    allowedMasks = ['StructuredElementL', 'Diagonal3x3x3', 'Diagonal10x10x4',
                          'Diagonal10x10x10']

    def __init__(self, mask='StructuredElementL', L=3):
        self.name = "MMFilterDil"
        self.mask = mask
        self.L = L
        self.maskImages = []

        self.clattr = None
        self.atts = None

    def toJSONString(self):
        pass

    def getName(self):
        return "MMFilterDil"

    def getInfo(self):
        info = helpers.FilterInfo()
        info.name = self.getName()
        info.memtype = bytes
        info.useTempBuffer = True
        # info.memtype = POCLFilter.POCLFilter.Type.Byte
        info.overlapX = info.overlapY = info.overlapZ = self.overlapAmount()
        return info

    def overlapAmount(self):
        # return number of slices in mask - how to determine this?
        pass

    def loadKernel(self):

        try:
            filename = "MMdil3D.cl"
            self.program = cl.Program(self.clattr.context, pkg.resource_string(__name__, filename)).build()
        except Exception:
            return False

        self.kernel = cl.Kernel(self.program, "MMdil3DFilterInit")
        self.kernel2 = cl.Kernel(self.program, "MMdil3DFilter")
        return True

    def runKernel(self, maskImages, overlapAmount):

        globalSize = [0, 0]
        localSize = [0, 0]
        self.clattr.computeWorkingGroupSize(localSize, globalSize, [self.atts.width, self.atts.height, 1])

        for i in range(maskImages.shape[0]):
            mask = maskImages[i]
            size = [0, 0, 0]
            size[2] = mask.shape[0]
            size[1] = mask.shape[1]
            size[0] = mask.shape[2]

            structElem = self.atts.getStructElement(self.clattr.context, self.clattr.queue, mask)
            startOffset = 0
            endOffset = 0

            if self.atts.overlap[self.index] > 0:
                startOffset = int(self.atts.overlap[self.index] / 2)
                endOffset = int(self.atts.overlap[self.index] / 2)

            if self.atts.sliceStart <= 0:
                startOffset = 0
            if self.atts.sliceEnd >= 0:
                endOffset = 0

            if i == 0:
                self.kernel.set_args(self.clattr.inputBuffer, self.clattr.outputTmpBuffer, np.int32(self.atts.width),
                                     np.int32(self.atts.height),
                                     np.int32(self.clattr.maxSliceCount+self.atts.overlap[self.index]),
                                     structElem, np.int32(size[0]), np.int32(size[1]), np.int32(size[2]),
                                     np.int32(startOffset), np.int32(endOffset))
            else:
                tmpBuffer1 = self.clattr.outputTmpBuffer if i%2 != 0 else self.clattr.outputBuffer
                tmpBuffer2 = self.clattr.outputTmpBuffer if i%2 == 0 else self.clattr.outputBuffer

                self.kernel2.set_args(tmpBuffer1, tmpBuffer2, np.int32(self.atts.width), np.int32(self.atts.height),
                                      np.int32(self.clattr.maxSliceCount + self.atts.overlap[self.index]),
                                      structElem, np.int32(size[0]), np.int32(size[1]), np.int32(size[2]),
                                      np.int32(startOffset), np.int32(endOffset))

                try:
                    cl.enqueue_nd_range_kernel(self.clattr.queue, self.kernel, globalSize, localSize)

                    cl.enqueue_nd_range_kernel(self.clattr.queue, self.kernel if i ==0 else self.kernel2,
                                               globalSize, localSize)
                except Exception:
                    return False

                structElem.release()

        if maskImages.shape[0]%2 != 0:
            tmpBuffer = self.clattr.outputTmpBuffer
            self.clattr.outputTmpBuffer = self.clattr.outputBuffer
            self.clattr.outputBuffer = tmpBuffer

        return True

    def runFilter(self):

        # TODO: check if mask is valid - probably needs similar machinery to self.overlapAmount

        filter_time = time.time()
        self.unKernel(self.maskImages, self.overlapAmount());

        cl.enqueue_copy(self.clattr.queue, self.clattr.inputBuffer, self.clattr.outputBuffer)
        filter_time = time.time() - filter_time

        return True



def setAttributes(self, CLAttributes, atts, index):
        self.clattr = CLAttributes
        self.atts = atts
        self.index = index
