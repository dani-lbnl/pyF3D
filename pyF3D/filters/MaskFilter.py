import numpy as np
import pkg_resources as pkg
import pyopencl as cl
import time
from pyF3D import helpers

class MaskFilter:

    allowedMasks = ['StructuredElementL', 'Diagonal3x3x3', 'Diagonal10x10x4',
                          'Diagonal10x10x10']
    maskChoices = ['mask3D']

    def __init__(self, maskChoice='mask3D', mask='structuredElementL', L=3):
        # Note: ignore 'L' parameter is mask choice is not 'structuredElementL'

        self.name = 'MaskFilter'

        self.maskChoice = maskChoice
        self.mask = mask
        self.L = L
        self.clattr = None
        self.atts = None

    def toJSONString(self):
        result = "{ \"Name\" : \"" + self.getName() + "\" , "
        result += "\"selectedMaskChoice\" : \"" + self.maskChoice + "\" , "

        mask = {"maskImage" : self.mask}
        if self.mask == 'StructuredElementL':
            mask["maskLen"] = "{}".format(int(self.L))

        result += "\"Mask\" : " + "{}".format(mask) + " }"
        return result

    def getName(self):
        return "MaskFilter"

    def getInfo(self):
        info = helpers.FilterInfo()
        info.name = self.getName()
        info.memtype = bytes
        # info.memtype = POCLFilter.POCLFilter.Type.Byte
        info.overlapX = info.overlapY = info.overlapZ = 0
        return info

    def loadKernel(self):
        try:
            filename = "Mask3D.cl"
            self.program = cl.Program(self.clattr.context, pkg.resource_string(__name__, filename)).build()
        except Exception:
            return False

        self.kernel = cl.Kernel(self.program, self.maskChoice)
        return True

    def runFilter(self):

        # probably will need to change this - use filteringattributes method
        mask = self.atts.maskImages[0]

        if self.atts.width*self.atts.height*self.atts.slices != mask.shape[0]*mask.shape[1].mask.shape[2]:
            print "Mask dimensions not equal to original image's"
            return False

        filter_time = time.time()
        globalSize = [0]
        localSize = [0]

        self.clattr.computeWorkingGroupSize(localSize, globalSize, [self.atts.width, self.atts.height,
                                                self.clattr.maxSliceCount + self.atts.overlap[self.index]])
        self.maskBuffer = self.atts.getStructElement(self.clattr.context, self.clattr.queue, mask, globalSize[0])

        try:
            self.kernel.set_args(self.clattr.inputBuffer, self.maskBuffer, self.clattr.outputBuffer,
                                 np.int32(self.atts.width), np.int32(self.atts.height),
                                 np.int32(self.clattr.maxSliceCount + self.atts.overlap[self.index]))

            # TODO: worry about copying data here? as is done in JOCL filters?

            cl.enqueue_nd_range_kernel(self.clattr.queue, self.kernel, globalSize, localSize)

        except Exception as e:
            raise e

            # write results
        cl.enqueue_copy(self.clattr.queue, self.clattr.inputBuffer, self.clattr.outputBuffer)
        self.clattr.queue.finish()

        filter_time = time.time() - filter_time

        return True

    def releaseKernel(self):

        if self.maskBuffer: del(self.maskBuffer)
        if self.kernel: del(self.kernel)
        return True

    def setAttributes(self, CLAttributes, atts, index):
        self.clattr = CLAttributes
        self.atts = atts
        self.index = index



