import numpy as np
import pyopencl as cl
import re

class FilteringAttributes:

    def __init__(self):
        self.width = 0
        self.height = 0
        self.channels = 0
        self.slices = 0
        self.sliceStart = 0
        self.sliceEnd = 0
        self.maxOverlap = 0
        self.overlap = []
        self.intermediateSteps = False
        self.chooseConstantDevices = False
        self.inputDeviceLength = 1

        self.MAX_STRUCTELEM_SIZE = 21*21*21
        self.internalImages = ["StructuredElementL", "Diagonal3x3x3", "Diagonal10x10x4", "Diagonal10x10x10"]

    def getMaskImages(self, maskImage, maskL):
        images = []
        if type(maskImage) is str:
            self.parseImage(maskImage, maskL, images)
        else:
            images.append(maskImage)
        return images

    def parseImage(self, inputString, maskL, images):

        if len(inputString) == 0:
            return False

        if inputString.startswith('StructuredElementL'):
            if images is not None:
                for image in self.buildStructElementArray(maskL):
                    images.append(image)
            return True
        elif re.match("Diagonal\d{1,2}x\d{1,2}x\d{1,2}", inputString):
            matches = re.findall("\d{1,2}", inputString)
            x = int(matches[0])
            y = int(matches[1])
            z = int(matches[2])

            if images is not None:
                images.append(self.buildDiagonalImage(x, y, z))
        else:
            raise TypeError('Mask type not identifiable')



    def buildStructElementArray(self, L):

        images = []
        S = 1
        middle = int(np.floor(L/2))

        # type 1, 2, 3
        stack = np.zeros((S, L, S)).astype(np.uint8)
        images.append(stack)

        stack = np.zeros((S, S, L)).astype(np.uint8)
        images.append(stack)

        stack = np.zeros((L, S, S)).astype(np.uint8)
        images.append(stack)

        # diagonals
        stack = np.zeros((L, L, L)).astype(np.uint8)
        processor = stack[middle]
        for j in range(L):
            processor[j, j] = 255
        images.append(stack)

        stack  = np.zeros((L, L, L)).astype(np.uint8)
        processor = stack[middle]
        for j in range(L):
            processor[j, L-j-1] = 255
        images.append(stack)

        stack = np.zeros((L, L, L)).astype(np.uint8)
        for j in range(L):
            processor = stack[j]
            processor[j, j] = 255
        images.append(stack)

        # top left into bottom right
        stack = np.zeros((L, L, L)).astype(np.uint8)
        for j in range(L):
            processor = stack[j]
            processor[j, L-j-1] = 255
        images.append(stack)

        #bottom left into top right
        stack = np.zeros((L, L, L)).astype(np.uint8)
        for j in range(L):
            processor = stack[j]
            processor[j, L-j-1] = 255
        images.append(stack)

        # bottom right into top left
        stack = np.zeros((L, L, L)).astype(np.uint8)
        for j in range(L):
            processor = stack[j]
            processor[L-j-1, L-j-1] = 255
        images.append(stack)

        #top left into bottom left
        stack = np.zeros((L, L, L)).astype(np.uint8)
        for j in range(L):
            processor = stack[j]
            processor[L-j-1, j] = 255
        images.append(stack)

        return images

    def buildDiagonalImage(self, width, height, slices):
        stack = np.zeros((int(slices), int(height), int(width))).astype(np.uint8)
        for i in range(int(slices)):
            prc = stack[i]
            endIndex = int(width) if int(width)<int(height) else int(height)
            for j in range(endIndex):
                prc[j, j] = 255
        return stack

    # test if mask is valid
    def isValidStructElement(self, image):
        if image.shape[0]*image.shape[1]*image.shape[2] >= self.MAX_STRUCTELEM_SIZE:
            return False
        return True

    def getStructElement(self, context, queue, stack, overrideSize=-1):

        # stack is np.ndarray - needs defined dimensions, so how does StructuredElementL work?

        # if not self.isValidStructElement(stack):
        #     return None
        size = [0, 0, 0]
        size[2] = stack.shape[0]
        size[1] = stack.shape[1]
        size[0] = stack.shape[2]

        if overrideSize >= np.product(size):
            structElem = cl.Buffer(context, cl.mem_flags.READ_WRITE, overrideSize*8)
        else:
            structElem = cl.Buffer(context, cl.mem_flags.READ_WRITE, np.product(size)*8)

        cl.enqueue_copy(queue, structElem, stack)
        return structElem

