import numpy as np
import pkg_resources as pkg
import pyopencl as cl
import pyF3D.FilterClasses as fc

class BilateralFilter:

    """
    Class for a bilateral filter of specified spatial radius and specified range radius.

    Parameters
    ----------
    spatialRadius: int, optional
        Specifies spatial radius
    rangeRadius: int, optional
        Specifies range radius
    """

    def __init__(self, spatialRadius=3, rangeRadius=30):

        self.name = 'BilateralFilter'

        self.spatialRadius = spatialRadius
        self.rangeRadius = rangeRadius
        self.clattr = None
        self.atts = None

    def toJSONString(self):
        result = "{ \"Name\" : \"" + self.getName() + "\" , "
        result += "\"spatialRadius\" : \"" + str(self.spatialRadius) + "\" , "
        result += "\"rangeRadius\" : \"" + str(self.rangeRadius) + "\" }"

    def clone(self):
        return BilateralFilter(spatialRadius=self.spatialRadius, rangeRadius=self.rangeRadius)

    def setSpatialRadius(self, sRadius):
        try:
            sRadius = int(sRadius)
            self.spatialRadius = sRadius
        except ValueError:
            raise ValueError('spatialRadius must be int')

    def setRangeRadius(self, rRadius):
        try:
            rRadius = int(rRadius)
            self.rangeRadius = rRadius
        except ValueError:
            raise ValueError('rangeRadius must be int')

    def getSpatialRadius(self):
        return self.spatialRadius

    def getRangeRadius(self):
        return self.rangeRadius

    # necessary?
    def getOptions(self):
        return "{}"

    def getInfo(self):
        info = fc.FilterInfo()
        info.name = self.getName()
        info.memtype = bytes
        # info.memtype = POCLFilter.POCLFilter.Type.Byte
        info.overlapX = self.spatialRadius
        info.overlapY = self.spatialRadius
        info.overlapZ = self.spatialRadius
        return info


    def getName(self):
        return "BilateralFilter"

    def makeKernel(self, r):

        radius = r + 1
        minWorkingGroup = 256
        if 'CPU' in self.clattr.device.name: minWorkingGroup = 64

        bufferSize = radius**2 - 1
        localSize = min(self.clattr.device.max_work_group_size, minWorkingGroup)
        globalSize = self.clattr.roundUp(localSize, bufferSize)*np.float32(0).nbytes

        buffer = cl.Buffer(self.clattr.context, cl.mem_flags.READ_WRITE, size = globalSize)
        kernel = cl.Kernel(self.program, 'makeKernel')
        kernel.set_args(np.float32(radius), buffer, np.int32(bufferSize))

        globalSize = [int(globalSize)]
        localSize = [int(localSize)]

        cl.enqueue_nd_range_kernel(self.clattr.queue, kernel, globalSize, localSize)
        output = np.empty(bufferSize).astype(np.float32)
        cl.enqueue_copy(self.clattr.queue, output, buffer)
        self.clattr.queue.finish()

        total = np.float32(0)
        for i in range(bufferSize):
            total += output[i]

        normalizeKernel = cl.Kernel(self.program, 'normalizeKernel')
        normalizeKernel.set_args(np.float32(total), buffer, np.int32(bufferSize))
        cl.enqueue_nd_range_kernel(self.clattr.queue, normalizeKernel, globalSize, localSize)


        return buffer


    def loadKernel(self):
        try:
            filename = "../OpenCL/BilateralFiltering.cl"
            self.program = cl.Program(self.clattr.context, pkg.resource_string(__name__, filename).decode()).build()
        except Exception:
            return  False

        self.spatialKernel = self.makeKernel(self.spatialRadius)
        self.rangeKernel = self.makeKernel(self.rangeRadius)
        self.kernel = cl.Kernel(self.program, 'BilateralFilter')
        return True

    def runFilter(self):

        globalSize = [0, 0]
        localSize = [0, 0]
        self.clattr.computeWorkingGroupSize(localSize, globalSize, [self.atts.width, self.atts.height, 1])

        try:
            self.kernel.set_args(self.clattr.inputBuffer, self.clattr.outputBuffer, np.int32(self.atts.width),
                                 np.int32(self.atts.height), np.int32(self.clattr.maxSliceCount + self.getInfo().overlapZ),
                                 self.spatialKernel, np.int32((self.spatialRadius+1)*2 - 1),
                                 self.rangeKernel, np.int32((self.rangeRadius+1)*2 - 1))

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










