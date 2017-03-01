import numpy as np
import pkg_resources as pkg
import pyopencl as cl
import time
from pyF3D import helpers

class BilateralFilter:

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
        info = helpers.FilterInfo()
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
        localSize = min(self.clattr.device.)













