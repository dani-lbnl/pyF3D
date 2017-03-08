import numpy as np
import pkg_resources as pkg
import pyopencl as cl
import time
from pyF3D import helpers
import MMFilterEro as mmero
import MMFilterDil as mmdil
import re

class MMFilterOpe:

    allowedMasks = ['StructuredElementL', 'Diagonal3x3x3', 'Diagonal10x10x4',
                          'Diagonal10x10x10']

    def __init__(self, mask='StructuredElementL', L=3):
        self.name = "MMFilterOpe"
        self.mask = mask
        self.L = L

        self.clattr = None
        self.atts = None

    def toJSONString(self):
        result = "{ \"Name\" : \"" + self.getName() + "\" , "
        mask = {"maskImage": self.mask}
        if self.mask == 'StructuredElementL':
            mask["maskLen"] = "{}".format(int(self.L))

        result += "\"Mask\" : " + "{}".format(mask) + " }"
        return result

    def getInfo(self):
        info = helpers.FilterInfo()
        info.name = self.getName()
        info.memtype = bytes
        # info.memtype = POCLFilter.POCLFilter.Type.Byte
        info.overlapX = info.overlapY = info.overlapZ = self.overlapAmount()
        return info

    def overlapAmount(self):

        if self.mask in self.allowedMasks:
            if self.mask.startswith('StructuredElement'):
                return self.L
            else:
                matches = re.findall("\d{1,2}", self.mask)
                return int(matches[-1])
        else:
            pass
            # figure out what to do with custom masks

    def getName(self):
        return "MMFilterOpe"

    def loadKernel(self):

        self.dilation = mmdil.MMFilterDil(mask=self.mask, L=self.L)
        self.erosion = mmero.MMFilterEro(mask=self.mask, L=self.L)

        self.dilation.setAttributes(self.clattr, self.atts, self.index)
        if not self.dilation.loadKernel():
            return False

        self.erosion.setAttributes(self.clattr, self.atts, self.index)
        if not self.erosion.loadKernel():
            return False

        return True

    def runFilter(self):

        maskImages = self.atts.getMaskImages(self.mask, self.L)
        for mask in maskImages:
            if not self.atts.isValidStructElement(mask):
                print "ERROR: Structure element size is too large..."
                return False

        filter_time = time.time()

        if not self.erosion.runKernel(maskImages, self.overlapAmount()):
            return False

        # swap results to put output back to input
        tmpBuffer = self.clattr.inputBuffer
        self.clattr.inputBuffer = self.clattr.outputBuffer
        self.clattr.outputBuffer = tmpBuffer

        if not self.dilation.runKernel(maskImages, self.overlapAmount()):
            return False

        cl.enqueue_copy(self.clattr.queue, self.clattr.inputBuffer, self.clattr.outputBuffer)
        filter_time = time.time() - filter_time

        return True

    def releaseKernel(self):
        if self.dilation: del (self.dilation)
        if self.erosion: del (self.erosion)
        return True

    def setAttributes(self, CLAttributes, atts, index):
            self.clattr = CLAttributes
            self.atts = atts
            self.index = index