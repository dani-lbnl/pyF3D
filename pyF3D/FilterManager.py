import pyopencl as cl
import time

import concurrent.futures as cf
import numpy as np

from . import ClAttributes
from . import FilterAttributes
from .filters import MedianFilter as mf
from .filters import FFTFilter as fft
from .filters import BilateralFilter as bf
from .filters import MaskFilter as mskf
from .filters import MMFilterDil as mmdil
from .filters import MMFilterEro as mmero
from .filters import MMFilterClo as mmclo
from .filters import MMFilterOpe as mmope
from . import FilterClasses as fc
import threading

startIndex = 0

def run_f3d(image, pipeline, platform=None):
    """
    Perform F3D filtering on image with specified pipeline

    Parameters
    ----------
    image: ndarray
        3D image data
    pipeline: list
        series of functions to be performed on image
    platform: {pyopencl.Platform, list, dict}, optional
        Platforms on which calculations are performed. Can either specify:

        1). A pyopencl.Platform object, for all calculations on single platform
        2). A list of pyopencl.Platform objects, for calculations to be performed in parallel
        3). A dictionary of pyopencl.Platform and int key/value pairs. The int values specify the maximum number of
            slices to be placed on the platform at any time (ex.: {platform1: 100} will assign a maximum of 100 slices to
            platform1)

    Returns
    -------
    ndarray
        Filtered 3D object
    """

    image = scale_to_uint8(image)
    stacks = runPipeline(image, pipeline, platform=platform)
    return reconstruct_final_image(stacks)

def runPipeline(image, pipeline, platform=None):
    """

    Performs filters contained in pipeline on input image. Creates one thread per OpenCL platform

    Parameters
    ----------
    image: ndarray
        3D image data
    pipeline: list
        series of functions to be performed on image
    platform: {pyopencl.Platform, list, dict}, optional
        Platforms on which calculations are performed. Can either specify:

        1). A pyopencl.Platform object, for all calculations on single platform
        2). A list of pyopencl.Platform objects, for calculations to be performed in parallel
        3). A dictionary of pyopencl.Platform and int key/value pairs. The int values specify the maximum number of
            slices to be placed on the platform at any time (ex.: {platform1: 100} will assign a maximum of 100 slices to
            platform1)


    Returns
    -------
    list
        portions of filtered data. Must be sorted.
    """
    global startIndex
    startIndex = 0
    stacks = []
    platform = check_if_valid_platform(platform)

    atts = FilterAttributes.FilteringAttributes()
    atts.overlap = [0]*len(platform)
    jobs = []
    with cf.ThreadPoolExecutor(len(platform)) as e:
        for i in range(len(platform)):
            index = i
            if type(platform) is list:
                p = platform[i]
                maxSliceCount = None
            else:
                p = list(platform.keys())[i]
                maxSliceCount = list(platform.values())[i]
            kwargs = {'image': image, 'pipeline': pipeline, 'attr': atts, 'platform': p,
                      'sliceCount': maxSliceCount, 'index': index, 'stacks': stacks}
            jobs.append(e.submit(doFilter, **kwargs))
            # e.submit(doFilter, **kwargs)
            # doFilter(**kwargs) #debug
    for job in jobs:
        job.result()
    return stacks

def doFilter(image, pipeline, attr, platform, sliceCount, index, stacks):

    global startIndex

    device = platform.get_devices()[0]
    device, context, queue = setup_cl_prereqs(device)
    clattr = ClAttributes.ClAttributes(context, device, queue, None, None, None)
    clattr.setMaxSliceCount(image, sliceCount)


    maxOverlap = 0
    for filter in pipeline:
        maxOverlap = max(maxOverlap, filter.getInfo().overlapZ)

    maxSliceCount = clattr.maxSliceCount
    clattr.initializeData(image, attr, maxOverlap, maxSliceCount)

    for filter in pipeline:
        if filter.getInfo().useTempBuffer:
            clattr.outputTmpBuffer = cl.Buffer(clattr.context, cl.mem_flags.READ_WRITE, clattr.inputBuffer.size)
            break
    stackRange = [0, 0]
    while True:

        if startIndex >= image.shape[0]:
        #if startIndex >= image.shape[2]:
            break
        getNextRange(image, stackRange, maxSliceCount, maxOverlap)
        attr.sliceStart = stackRange[0]
        attr.sliceEnd = stackRange[1]
        clattr.loadNextData(image, attr, stackRange[0], stackRange[1], maxOverlap)
        attr.overlap[index] = maxOverlap
        pipelineTime = 0
        for i in range(len(pipeline)):

            filter = pipeline[i].clone()
            filter.setAttributes(clattr, attr, index)

            if not filter.loadKernel():
                raise Exception

            filterTime = time.time()

            if not filter.runFilter():
                raise Exception

            filterTime = time.time() - filterTime
            pipelineTime += filterTime

            if i < len(pipeline) - 1:
                clattr.swapBuffers()

        result = clattr.writeNextData(attr, stackRange[0], stackRange[1], maxOverlap)
        addResultStack(stacks, stackRange[0], stackRange[1], result, clattr.device.name, pipelineTime)


    clattr.inputBuffer.release()
    clattr.outputBuffer.release()
    if clattr.outputTmpBuffer is not None:
        clattr.outputTmpBuffer.release()

    return stacks

def run_MedianFilter(image, platform=None):
    """
    Performs median filter (radius = 3) on image

    Parameters
    ----------
    image: ndarray
        3D image data
    platform: {pyopencl.Platform, list, dict}, optional
        Platforms on which calculations are performed. Can either specify:

        1). A pyopencl.Platform object, for all calculations on single platform
        2). A list of pyopencl.Platform objects, for calculations to be performed in parallel
        3). A dictionary of pyopencl.Platform and int key/value pairs. The int values specify the maximum number of
            slices to be placed on the platform at any time (ex.: {platform1: 100} will assign a maximum of 100 slices to
            platform1)


    Returns
    -------
    ndarray
        3D image after median filtering
    """

    pipeline = [mf.MedianFilter()]
    stacks = runPipeline(image, pipeline, platform=platform)
    return reconstruct_final_image(stacks)


def run_FFTFilter(image, FFTChoice='Forward', platform=None):
    """
    Performs FFT filter on image

    Parameters
    ----------
    image: ndarray
        3D image data
    FFTChoice: str, optional
        Either 'Forward' or 'Inverse'
    platform: {pyopencl.Platform, list, dict}, optional
        Platforms on which calculations are performed. Can either specify:

        1). A pyopencl.Platform object, for all calculations on single platform
        2). A list of pyopencl.Platform objects, for calculations to be performed in parallel
        3). A dictionary of pyopencl.Platform and int key/value pairs. The int values specify the maximum number of
            slices to be placed on the platform at any time (ex.: {platform1: 100} will assign a maximum of 100 slices to
            platform1)

    Returns
    -------
    ndarray
        3D image after FFT filtering
    """

    pipeline = [fft.FFTFilter(FFTChoice=FFTChoice)]
    stacks = runPipeline(image, pipeline, platform=platform)
    return reconstruct_final_image(stacks)

def run_BilateralFilter(image, spatialRadius=3, rangeRadius=30, platform=None):
    """
    Performs bilateral filter on image

    Parameters
    ----------
    image: ndarray
        3D image data
    spatialRadius: int, optional
        Specifies spatial radius
    rangeRadius: int, optional
        Specifies range radius
    platform: {pyopencl.Platform, list, dict}, optional
        Platforms on which calculations are performed. Can either specify:

        1). A pyopencl.Platform object, for all calculations on single platform
        2). A list of pyopencl.Platform objects, for calculations to be performed in parallel
        3). A dictionary of pyopencl.Platform and int key/value pairs. The int values specify the maximum number of
            slices to be placed on the platform at any time (ex.: {platform1: 100} will assign a maximum of 100 slices to
            platform1)


    Returns
    -------
    ndarray
        3D image after bilateral filtering
    """


    pipeline = [bf.BilateralFilter(spatialRadius=spatialRadius, rangeRadius=rangeRadius)]
    stacks = runPipeline(image, pipeline, platform=platform)
    return reconstruct_final_image(stacks)


def run_MaskFilter(image, maskChoice='mask3D', mask='StructuredElementL', L=3, platform=None):

    """
    Performs mask filter on image

    Parameters
    ----------
    image: ndarray
        3D image data
    maskChoice: str, optional
        type of mask - can only be 'mask3D' currently
    mask: {sr, ndarray}, optional
        Mask must be same shape as image. Can be one of the following string values:

        'StructuredElementL'
        ''Diagonal3x3x3'
        ''Diagonal10x10x4'
        ''Diagonal10x10x10'

        Can also be ndarray that will be used directly as a mask
    L: int, optional
        Radius for 'StructuredElementL'
    platform: {pyopencl.Platform, list, dict}, optional
        Platforms on which calculations are performed. Can either specify:

        1). A pyopencl.Platform object, for all calculations on single platform
        2). A list of pyopencl.Platform objects, for calculations to be performed in parallel
        3). A dictionary of pyopencl.Platform and int key/value pairs. The int values specify the maximum number of
            slices to be placed on the platform at any time (ex.: {platform1: 100} will assign a maximum of 100 slices to
            platform1)

    """
    pipeline = [mskf.MaskFilter(maskChoice=maskChoice, mask=mask, L=L)]
    stacks = runPipeline(image, pipeline, platform=platform)
    return reconstruct_final_image(stacks)


def run_MMFilterDil(image, mask='StructuredElementL', L=3, platform=None):
    """
    Performs dilation filter on image

    Parameters
    ----------
    image: ndarray
        3D image data
    mask: {str, ndarray}, optional
        Must be one of the following string values:

        'StructuredElementL'
        ''Diagonal3x3x3'
        ''Diagonal10x10x4'
        ''Diagonal10x10x10'

        Can also be ndarray that will be used directly as a mask
    L: int, optional
        Radius for 'StructuredElementL'
    platform: {pyopencl.Platform, list, dict}, optional
        Platforms on which calculations are performed. Can either specify:

        1). A pyopencl.Platform object, for all calculations on single platform
        2). A list of pyopencl.Platform objects, for calculations to be performed in parallel
        3). A dictionary of pyopencl.Platform and int key/value pairs. The int values specify the maximum number of
            slices to be placed on the platform at any time (ex.: {platform1: 100} will assign a maximum of 100 slices to
            platform1)


    Returns
    -------
    ndarray
        3D image after dilation filtering
    """

    pipeline = [mmdil.MMFilterDil(mask=mask, L=L)]
    stacks = runPipeline(image, pipeline, platform=platform)
    return reconstruct_final_image(stacks)


def run_MMFilterEro(image, mask="StructuredElementL", L=3, platform=None):
    """
    Performs erosion filter on image

    Parameters
    ----------
    image: ndarray
        3D image data
    mask: {str, ndarray}, optional
        Must be one of the following string values:

        'StructuredElementL'
        ''Diagonal3x3x3'
        ''Diagonal10x10x4'
        ''Diagonal10x10x10'

        Can also be ndarray that will be used directly as a mask
    L: int, optional
        Radius for 'StructuredElementL'
    platform: {pyopencl.Platform, list, dict}, optional
        Platforms on which calculations are performed. Can either specify:

        1). A pyopencl.Platform object, for all calculations on single platform
        2). A list of pyopencl.Platform objects, for calculations to be performed in parallel
        3). A dictionary of pyopencl.Platform and int key/value pairs. The int values specify the maximum number of
            slices to be placed on the platform at any time (ex.: {platform1: 100} will assign a maximum of 100 slices to
            platform1)


    Returns
    -------
    ndarray
        3D image after erosion filtering
    """

    pipeline = [mmero.MMFilterEro(mask=mask, L=L)]
    stacks = runPipeline(image, pipeline, platform=platform)
    return reconstruct_final_image(stacks)


def run_MMFilterClo(image, mask='StructuredElementL', L=3, platform=None):
    """
    Performs closing filter on image

    Parameters
    ----------
    image: ndarray
        3D image data
    mask: {str, ndarray}, optional
        Must be one of the following string values:

        'StructuredElementL'
        ''Diagonal3x3x3'
        ''Diagonal10x10x4'
        ''Diagonal10x10x10'

        Can also be ndarray that will be used directly as a mask
    L: int, optional
        Radius for 'StructuredElementL'
    platform: {pyopencl.Platform, list, dict}, optional
        Platforms on which calculations are performed. Can either specify:

        1). A pyopencl.Platform object, for all calculations on single platform
        2). A list of pyopencl.Platform objects, for calculations to be performed in parallel
        3). A dictionary of pyopencl.Platform and int key/value pairs. The int values specify the maximum number of
            slices to be placed on the platform at any time (ex.: {platform1: 100} will assign a maximum of 100 slices to
            platform1)


    Returns
    -------
    ndarray
        3D image after closing filtering
    """

    pipeline = [mmclo.MMFilterClo(mask=mask, L=L)]
    stacks = runPipeline(image, pipeline, platform=platform)
    return reconstruct_final_image(stacks)


def run_MMFilterOpe(image, mask='StructuredElementL', L=3, platform=None):
    """
    Performs opening filter on image

    Parameters
    ----------
    image: ndarray
        3D image data
    mask: {str, ndarray}, optional
        Must be one of the following string values:

        'StructuredElementL'
        ''Diagonal3x3x3'
        ''Diagonal10x10x4'
        ''Diagonal10x10x10'

        Can also be ndarray that will be used directly as a mask
    L: int, optional
        Radius for 'StructuredElementL'
    platform: {pyopencl.Platform, list, dict}, optional
        Platforms on which calculations are performed. Can either specify:

        1). A pyopencl.Platform object, for all calculations on single platform
        2). A list of pyopencl.Platform objects, for calculations to be performed in parallel
        3). A dictionary of pyopencl.Platform and int key/value pairs. The int values specify the maximum number of
            slices to be placed on the platform at any time (ex.: {platform1: 100} will assign a maximum of 100 slices to
            platform1)


    Returns
    -------
    ndarray
        3D image after opening filtering
    """

    pipeline = [mmope.MMFilterOpe(mask=mask, L=L)]
    stacks = runPipeline(image, pipeline, platform=platform)
    return reconstruct_final_image(stacks)

def reconstruct_final_image(stacks):
        stacks = sorted(stacks)
        image = stacks[0].stack

        for stack in stacks[1:]:
            image = np.append(image, stack.stack, axis=0)

        return image

load_lock = threading.Lock()
def getNextRange(image, range, sliceCount, overlap):

    global startIndex

    with load_lock:
        endIndex = image.shape[0]
        #endIndex = image.shape[2]
        range[0] = max(0, startIndex)
        range[1] = max(0, startIndex - overlap) + sliceCount - overlap
        if range[1] >= endIndex:
            range[1] = endIndex
        startIndex = range[1]
        return startIndex

result_lock = threading.Lock()
def addResultStack(stacks, startRange, endRange, output, name, pipelineTime):

    with result_lock:
        sr = fc.StackRange()
        sr.startRange = startRange
        sr.endRange = endRange
        sr.stack = output
        sr.time = pipelineTime
        sr.name = name

        stacks.append(sr)

def check_if_valid_platform(platform=None):
    if not platform:
        platform = [cl.get_platforms()[0]]
    if type(platform) is not list and type(platform) is not dict:
        platform = [platform]

    if type(platform) is list:
        for item in platform:
            if type(item) is not cl.Platform:
                raise TypeError("\'platform\' argument must be of type pyopencl.Platform")
    else: # platform is dict
        try:
            for key, val in platform.iteritems():
                if type(key) is not cl.Platform:
                    raise TypeError("\'platform\' argument must be of type pyopencl.Platform")
                try:
                    platform[key] = int(val)
                except ValueError:
                    raise TypeError('Values must be convertable to int')
        except:
            for key, val in platform.items():
                if type(key) is not cl.Platform:
                    raise TypeError("\'platform\' argument must be of type pyopencl.Platform")
                try:
                    platform[key] = int(val)
                except ValueError:
                    raise TypeError('Values must be convertable to int')
    return platform


def setup_cl_prereqs(device=None):
    context = cl.Context([device])
    queue = cl.CommandQueue(context, device)

    return device, context, queue

def scale_to_uint8(data):
    """
    Scales input array to np.uint8 type

    Parameters
    ----------
    data: np.ndarray
        Input data. If not type np.ndarray, must be able to convert to np.ndarray

    Returns
    -------
    np.ndarray
        data as type np.uint8 (8-bit)
    """
    if type(data) is np.ndarray and data.dtype is np.dtype('uint8'):
        return data
    else:
        if type(data) is not np.ndarray:
            data = np.array(data)

        a = float(255)/(float(np.max(data)) - float(np.min(data)))
        b = (float(255)*float(np.min(data)))/(float(np.min(data)) - float(np.max(data)))

        data = a*data + b
        return (data).astype(np.uint8)

