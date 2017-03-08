import numpy as np
import pkg_resources as pkg
import pyopencl as cl
import time
from pyF3D import helpers


class MedianFilter:

    def __init__(self):
        self.name = 'MedianFilter'
        self.index = -1

        # load clattr from RunnablePOCLFilter
        self.clattr = None
        self.atts = None #load attributes from RunnablePOCLFilter

    def toJSONString(self):
        result = "{ \"Name\" : \"" + self.getName() + "\", " + "\" }"
        return result

    def getInfo(self):
        info = helpers.FilterInfo()
        info.name = self.getName()
        info.memtype = bytes
        # info.memtype = POCLFilter.POCLFilter.Type.Byte
        info.overlapX = info.overlapY = info.overlapZ = 0
        return info

    def getName(self):
        return "MedianFilter"

    def loadKernel(self):
        # median_comperror = ""

        print self.clattr.device, "1"

        try:
            filename = "MedianFilter.cl"
            print self.clattr.device, "2"

            program = cl.Program(self.clattr.context, pkg.resource_string(__name__, filename)).build()
            print self.clattr.device, "3"
        except Exception as e:
            raise e


            # other stuff to log errors

        self.kernel = cl.Kernel(program, "MedianFilter")
        print self.clattr.device, "4"

        return True

    def runFilter(self):
        start_time = time.time()

        if self.atts.height == 1 and self.atts.slices == 1:
            mid = 1
        elif self.atts.slices == 1:
            mid = 4
        else: mid = 13

        globalSize = [0, 0]
        localSize = [0, 0]
        self.clattr.computeWorkingGroupSize(localSize, globalSize, [self.atts.width, self.atts.height, 1])

        try:

            # TODO: does loading happen here or in a clattr function?
            # copy data to accelerator
            # self.clattr.queue.enqueue_copy(self.clattr.queue, self.clattr.inputBuffer, )

            # set up parameters
            self.kernel.set_args(self.clattr.inputBuffer, self.clattr.outputBuffer, np.int32(self.atts.width),
                                           np.int32(self.atts.height), np.int32(self.clattr.maxSliceCount),
                                            np.int32(mid))

            # execute kernel
            cl.enqueue_nd_range_kernel(self.clattr.queue, self.kernel, globalSize, localSize)

        except Exception as e:
            raise e

        # write results
        cl.enqueue_copy(self.clattr.queue, self.clattr.inputBuffer, self.clattr.outputBuffer)
        self.clattr.queue.finish()
        return True

    def releaseKernel(self):

        # Does garbage collector correctly free resources?

        if self.kernel:
            del(self.kernel)
        return True


    def setAttributes(self, CLAttributes, atts, index):
        self.clattr = CLAttributes
        self.atts = atts
        self.index = index








