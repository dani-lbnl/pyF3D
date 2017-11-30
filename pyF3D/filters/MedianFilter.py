import numpy as np
import pkg_resources as pkg
import pyopencl as cl
import pyF3D.FilterClasses as fc
import os
import sys

currdir = os.path.dirname(os.path.realpath(__file__))

class MedianFilter:
    """
    Class for a 3D median filter of radius 3 (no parameters)
    """

    def __init__(self):
        self.name = 'MedianFilter'
        self.index = -1

        self.clattr = None
        self.atts = None
        
    def clone(self):
        return MedianFilter()

    def toJSONString(self):
        result = "{ \"Name\" : \"" + self.getName() + "\", " + "\" }"
        return result

    def getInfo(self):
        info = fc.FilterInfo()
        info.name = self.getName()
        info.memtype = bytes
        info.overlapX = info.overlapY = info.overlapZ = 4
        return info

    def getName(self):
        return "MedianFilter"

    def loadKernel(self):
        try:
            filename = "../OpenCL/MedianFilter.cl"
            program = cl.Program(self.clattr.context, pkg.resource_string(__name__, filename).decode()).build()
        except Exception as e:
            raise e


        self.kernel = cl.Kernel(program, "MedianFilter")

        return True

    def runFilter(self):

        if self.atts.height == 1 and self.atts.slices == 1:
            mid = 1
        elif self.atts.slices == 1:
            mid = 4
        else: mid = 13

        globalSize = [0, 0]
        localSize = [0, 0]
        self.clattr.computeWorkingGroupSize(localSize, globalSize, [self.atts.width, self.atts.height, 1])

        try:
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



    def setAttributes(self, CLAttributes, atts, index):
        self.clattr = CLAttributes
        self.atts = atts
        self.index = index








