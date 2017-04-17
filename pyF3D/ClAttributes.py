import pyopencl as cl
import sys
# from pipeline import msg
import numpy as np
import pkg_resources as pkg

class ClAttributes(object):

    def __init__(self, context, device, queue, inputBuffer, outputBuffer, outputTmpBuffer):
        super(ClAttributes, self).__init__()

        self.context = context
        self.device = device
        self.queue = queue

        self.inputBuffer = inputBuffer
        self.outputBuffer = outputBuffer
        self.outputTmpBuffer = outputTmpBuffer

        # multiply by 8 for 8bit images?
        self.globalMemSize = int(min(self.device.max_mem_alloc_size * 0.5, sys.maxsize >> 1))
        # self.globalMemSize = int(min(self.device.max_mem_alloc_size * 0.5, sys.maxsize >> 1))
        if 'CPU' in self.device.name:
            self.globalMemSize = int(min(self.globalMemSize, 10*1024*1024*8))

        self.maxSliceCount = 0

    def roundUp(self, groupSize, globalSize):
        r = globalSize % groupSize
        return globalSize if r ==0 else globalSize + groupSize - r

    def computeWorkingGroupSize(self, localSize, globalSize, sizes):
        if not localSize or not globalSize or not sizes:
            return False
        elif len(localSize) <= 0 or len(localSize) > 2 or len(globalSize) <= 0 or len(globalSize) > 2 or len(
                sizes) != 3:
            return False

        # set working group sizes
        dimensions = len(globalSize)
        if dimensions == 1:
            localSize[0] = self.device.max_work_group_size
            globalSize[0] = self.roundUp(localSize[0], sizes[0]*sizes[1]*sizes[2])
        elif dimensions == 2:
            localSize[0] = min(int(np.sqrt(self.device.max_work_group_size)), 16)
            globalSize[0] = self.roundUp(localSize[0], sizes[0])

            localSize[1] = min(int(np.sqrt(self.device.max_work_group_size)), 16)
            globalSize[1] = self.roundUp(localSize[1], sizes[1])


        return True

    def setMaxSliceCount(self, image, maxSlice=None):

        dim = image.shape
        maxSliceCount = int(self.globalMemSize/(dim[1]*dim[2]*8))


        if maxSliceCount > dim[0]:
            maxSliceCount = dim[0]
        if not maxSlice or maxSlice>maxSliceCount:
            self.maxSliceCount = maxSliceCount
        else:
            self.maxSliceCount = maxSlice

    def initializeData(self, image, atts, overlapAmount, maxSliceCount):
        """
        :param image: type is np.ndarray?
        :param atts: type: POCLFilter
        :param overlapAmount:
        :param maxSliceCount:
        :return:
        """

        dims = image.shape

        atts.width = dims[2]
        atts.height = dims[1]
        atts.slices = dims[0]
        atts.sliceStart = -1
        atts.sliceEnd = -1

        atts.channels = 1 # for greyscale

        if maxSliceCount <= 0:
            return False

        totalSize = (atts.width * atts.height * maxSliceCount)*8

        self.inputBuffer = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, totalSize)
        self.outputBuffer = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, totalSize)
        return True


    def loadNextData(self, image, atts, startRange, endRange, overlap):

        minIndex = max(0, startRange - overlap)
        maxIndex = min(atts.slices, endRange + overlap)

        im = image[minIndex:maxIndex, :, :]
        dim = im.shape
        im = np.reshape(im, dim[0]*dim[1]*dim[2])
        cl.enqueue_copy(self.queue, self.inputBuffer, im)
        return True

    def writeNextData(self, atts, startRange, endRange, overlap):
        startIndex = 0 if startRange==0 else overlap
        length = endRange - startRange
        output = np.empty((length + startIndex+overlap)*atts.width*atts.height).astype(np.uint8)
        cl.enqueue_copy(self.queue, output, self.outputBuffer)
        output = output.reshape((startIndex+length+overlap), atts.height, atts.width)
        output = output[startIndex:startIndex+length]
        return output

    def swapBuffers(self):

        tmpBuffer = self.inputBuffer
        self.inputBuffer = self.outputBuffer
        self.outputBuffer = tmpBuffer

def create_cl_attributes():

    context = cl.create_some_context()
    device = context.devices[0]
    queue = cl.CommandQueue(context, device)

    return context, device, queue

def list_all_cl_platforms():
    return cl.get_platforms()





