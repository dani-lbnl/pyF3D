import numpy as np
import pyopencl as cl
import pyF3D.FilterClasses as fc
from . import MMFilterEro as mmero
from . import MMFilterDil as mmdil
import re

class MMFilterClo:

    """
    Class for a closing filter with specified mask

    Parameters
    ----------
    mask: {str, ndarray}, optional
        Must be one of the following string values:

        'StructuredElementL'
        ''Diagonal3x3x3'
        ''Diagonal10x10x4'
        ''Diagonal10x10x10'

        Can also be ndarray that will be used directly as a mask
    L: int, optional
        Radius for 'StructuredElementL'
    """

    allowedMasks = ['StructuredElementL', 'Diagonal3x3x3', 'Diagonal10x10x4',
                          'Diagonal10x10x10']

    def __init__(self, mask='StructuredElementL', L=3):
        self.name = "MMFilterClo"
        self.mask = mask
        self.L = L

        self.clattr = None
        self.atts = None

        if type(self.mask) is str:
            if self.mask not in self.allowedMasks: raise TypeError('Mask does not match any of allowed choices')
        else:
            try:
                self.mask = np.array(self.mask).astype(np.uint8)
            except ValueError:
                raise TypeError('Mask must be able to be converted to np.uint8')

    def toJSONString(self):
        result = "{ \"Name\" : \"" + self.getName() + "\" , "
        mask = {"maskImage" : self.mask if self.mask in self.allowedMasks else 'customMask'}
        if self.mask == 'StructuredElementL':
            mask["maskLen"] = "{}".format(int(self.L))

        result += "\"Mask\" : " + "{}".format(mask) + " }"
        return result

    def clone(self):
        return MMFilterClo(mask=self.mask, L=self.L)

    def getInfo(self):
        info = fc.FilterInfo()
        info.name = self.getName()
        info.memtype = bytes
        info.useTempBuffer = True
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
            return self.mask.shape[0]



    def getName(self):
        return "MMFilterClo"

    def loadKernel(self):

        self.erosion = mmero.MMFilterEro(mask=self.mask, L=self.L)
        self.dilation = mmdil.MMFilterDil(mask=self.mask, L=self.L)

        self.erosion.setAttributes(self.clattr, self.atts, self.index)
        if not self.erosion.loadKernel():
            return False

        self.dilation.setAttributes(self.clattr, self.atts, self.index)
        if not self.dilation.loadKernel():
            return False

        return True

    def runFilter(self):

        maskImages = self.atts.getMaskImages(self.mask, self.L)
        #print(maskImages[0])
        for mask in maskImages:
            if not self.atts.isValidStructElement(mask):
                print("ERROR: Structure element size is too large...")
                return False

        if not self.dilation.runKernel(maskImages, self.overlapAmount()):
            print("Problem running dilation!")
            return False

        # swap results to put output back to input
        tmpBuffer = self.clattr.inputBuffer
        self.clattr.inputBuffer = self.clattr.outputBuffer
        self.clattr.outputBuffer = tmpBuffer

        if not self.erosion.runKernel(maskImages, self.overlapAmount()):
            print("Problem running erosion!")
            return False

        cl.enqueue_copy(self.clattr.queue, self.clattr.inputBuffer, self.clattr.outputBuffer)

        return True

    def setAttributes(self, CLAttributes, atts, index):
            self.clattr = CLAttributes
            self.atts = atts
            self.index = index
