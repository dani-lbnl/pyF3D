import pyopencl as cl
# import time
# import concurrent.futures as cf
# import ClAttributes
# import FilterAttributes
# import helpers
import filters.MedianFilter as mf
import warnings

class Pipeline:

    filter_types =  {'MedianFilter': mf}#, 'BilateralFilter': bf, 'MMFilterClo': mmclo, 'FFTFilter': fft,
                    # 'MaskFilter': mskf, 'MMFilterDil': mmdil, 'MMFilterEro': mmero, 'MMFilterOpe': mmope}

    def __init__(self):

        self.pipeline = []

    def add(self, filter):

        if filter.__class__ in self.filter_types.values():
            self.pipeline.append(filter)
        else:
            warnings.warn('Can only add filters of approved type (make this more informative)', Warning)