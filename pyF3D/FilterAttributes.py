import numpy as np


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

    def parseImage(self):
        pass

    def buildStructElementArray(self, L):

        images = []
        S = 1
        middle = int(np.floor(L/2)) + 1

        # type 1, 2, 3
        stack = np.empty((S, L, S)).astype(np.uint8)
        images.append(stack)

        stack = np.empty((S, S, L)).astype(np.uint8)
        images.append(stack)

        stack = np.empty((L, S, S)).astype(np.uint8)
        images.append(stack)

        # diagonals
        stack = np.empty((L, L, L)).astype(np.uint8)
        processor = stack[middle]
        for j in range(L):
            processor[j, j] = 255
        images.append(stack)
