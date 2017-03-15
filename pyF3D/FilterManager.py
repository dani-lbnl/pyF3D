import pyopencl as cl
import time

import concurrent.futures as cf
import numpy as np

import ClAttributes
import FilterAttributes
import filters.MedianFilter as mf
import filters.FFTFilter as fft
import filters.BilateralFilter as bf
import filters.MaskFilter as mskf
import filters.MMFilterDil as mmdil
import filters.MMFilterEro as mmero
import filters.MMFilterClo as mmclo
import filters.MMFilterOpe as mmope
import FilterClasses as fc
import threading

startIndex = 0

def run_f3d(image, pipeline, platform=None):
    """
    Parameters
    ----------
    image: ndarray
        3D image data
    pipeline: list
        series of functions to be performed on image
    platform: pyopencl.Platform, optional
        Platform on which calculations are performed

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
    platform: pyopencl.Platform, optional
        Platform on which calculations are performed

    Returns
    -------
    list
        portions of filtered data. Must be sorted.
    """

    stacks = []
    platform = check_if_valid_platform(platform)

    atts = FilterAttributes.FilteringAttributes()
    atts.overlap = [0]*len(platform)
    with cf.ThreadPoolExecutor(len(platform)) as e:
        for i in range(len(platform)):
            index = i
            kwargs = {'image': image, 'pipeline': pipeline, 'attr': atts, 'platform': platform[i],
                      'index': index, 'stacks': stacks}
            e.submit(doFilter, **kwargs)
            # doFilter(**kwargs) #debug

    return stacks

def doFilter(image, pipeline, attr, platform, index, stacks):

    global startIndex

    device = platform.get_devices()[0]
    device, context, queue = setup_cl_prereqs(device)
    clattr = ClAttributes.ClAttributes(context, device, queue, None, None, None)
    clattr.setMaxSliceCount(image)


    start = startIndex
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

        if start >= image.shape[0]:
            break
        start = getNextRange(image, stackRange, maxSliceCount)
        attr.sliceStart = stackRange[0]
        attr.sliceEnd = stackRange[1]
        clattr.loadNextData(image, attr, stackRange[0], stackRange[1], maxOverlap)
        maxSliceCount = stackRange[1] - stackRange[0]
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


    startIndex = 0
    return stacks

def run_MedianFilter(image, platform=None):
    """
    Performs median filter (radius = 3) on image

    Parameters
    ----------
    image: ndarray
        3D image data
    platform: pyopencl.Platform, optional
        Platform on which calculations are performed

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
    platform: pyopencl.Platform, optional
        Platform on which calculations are performed

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
    spatialRadius: int
        Specifies spatial radius
    rangeRadius: int
        Specifies range radius
    platform: pyopencl.Platform, optional
        Platform on which calculations are performed

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
    NOT WORKING

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
    mask: {str, ndarray}
        Must be one of the following string values:

        'StructuredElementL'
        ''Diagonal3x3x3'
        ''Diagonal10x10x4'
        ''Diagonal10x10x10'

        Can also be ndarray that will be used directly as a mask
    L: int
        Radius for 'StructuredElementL'
    platform: pyopencl.Platform, optional
        Platform on which calculations are performed

    Returns
    -------
    ndarray
        3D image after dilation filtering
    """

    pipeline = [mmdil.MMFilterDil(mask=mask,L=L)]
    stacks = run_f3d(image, pipeline, platform=platform)
    return reconstruct_final_image(stacks)


def run_MMFilterEro(image, mask="StructuredElementL", L=3, platform=None):
    """
    Performs erosion filter on image

    Parameters
    ----------
    image: ndarray
        3D image data
    mask: {str, ndarray}
        Must be one of the following string values:

        'StructuredElementL'
        ''Diagonal3x3x3'
        ''Diagonal10x10x4'
        ''Diagonal10x10x10'

        Can also be ndarray that will be used directly as a mask
    L: int
        Radius for 'StructuredElementL'
    platform: pyopencl.Platform, optional
        Platform on which calculations are performed

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
    mask: {str, ndarray}
        Must be one of the following string values:

        'StructuredElementL'
        ''Diagonal3x3x3'
        ''Diagonal10x10x4'
        ''Diagonal10x10x10'

        Can also be ndarray that will be used directly as a mask
    L: int
        Radius for 'StructuredElementL'
    platform: pyopencl.Platform, optional
        Platform on which calculations are performed

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
    mask: {str, ndarray}
        Must be one of the following string values:

        'StructuredElementL'
        ''Diagonal3x3x3'
        ''Diagonal10x10x4'
        ''Diagonal10x10x10'

        Can also be ndarray that will be used directly as a mask
    L: int
        Radius for 'StructuredElementL'
    platform: pyopencl.Platform, optional
        Platform on which calculations are performed

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

        # for stack in stacks:
            # print stack.endRange - stack.startRange

        for stack in stacks[1:]:
            image = np.append(image, stack.stack, axis=0)

        return image

load_lock = threading.Lock()
def getNextRange(image, range, sliceCount):

    global startIndex

    with load_lock:
        endIndex = image.shape[0]
        range[0] = startIndex
        range[1] = startIndex + sliceCount
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
    if type(platform) is not list:
        platform = [platform]

    for item in platform:
        if type(item) is not cl.Platform:
            raise TypeError("\'platform\' argument must be of type pyopencl.Platform")
    return platform


def setup_cl_prereqs(device=None):
    context = cl.Context([device])
    queue = cl.CommandQueue(context, device)

    return device, context, queue

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

