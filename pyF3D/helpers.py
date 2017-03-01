import numpy as np


def scale_to_uint8(data):
    """
    Custom function to scale data to 8-bit uchar for F3D plugin
    """
    if type(data) is not np.ndarray:
        data = np.array(data)

    a = float(255)/(float(np.max(data)) - float(np.min(data)))
    b = (float(255)*float(np.min(data)))/(float(np.min(data)) - float(np.max(data)))

    data = a*data + b
    return (data).astype(np.uint8)

class StackRange:

    def __init__(self):
        self.processTime = 0
        self.startRange = 0
        self.endRange = 0
        self.name = ""
        self.stack = None

    def __lt__(self, sr):
        # ascending
        return self.startRange - sr.startRange < 0

    def __eq__(self, other):
        if self is other:
            return True
        elif type(self) != type(other):
            return False
        else:
            return self.startRange==other.startRange and self.endRange==other.endRange


class FilterInfo(object):
    def __init__(self):
        self.name = ""
        self.L = -1
        self.overlapX = 0
        self.overlapY = 0
        self.overlapZ = 0
        self.memtype = bytes
        # self.memtype = POCLFilter.Type.Byte
        self.useTempBuffer = False