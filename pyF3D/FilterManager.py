# import POCLFilter
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
import helpers
import threading

startIndex = 0

def run_f3d(image, pipeline, device=None):
    stacks = runPipeline(image, pipeline, device=device)
    return reconstruct_final_image(stacks)

def runPipeline(image, pipeline, device=None):

    device, context, queue = setup_cl_prereqs(device=device)
    stacks = []

    # at this point we have a list of devices, contexts, and queues
    atts = FilterAttributes.FilteringAttributes()
    atts.overlap = len(device)*[0]
    with cf.ThreadPoolExecutor(len(device)) as e:
        for i in range(len(device)):
            dev = device[i]
            ctx = context[i]
            q = queue[i]
            index = i
            clattr = ClAttributes.ClAttributes(ctx, dev, q, None, None, None)
            clattr.setMaxSliceCount(image)
            kwargs = {'image': image, 'pipeline': pipeline, 'attr': atts, 'clattr': clattr,
                      'index': index, 'stacks': stacks}
            e.submit(doFilter, **kwargs)
            # doFilter(**kwargs) #debug

    # with cf.ThreadPoolExecutor(len(device)) as e:
    #     for i in range(len(device)):
    #         if i == 0:
    #             dev = device[i]
    #             ctx = context[i]
    #             q = queue[i]
    #             index = i
    #             clattr = ClAttributes.ClAttributes(ctx, dev, q, None, None, None)
    #             clattr.setMaxSliceCount(image)
    #             kwargs = {'image': image, 'pipeline': pipeline, 'attr': atts, 'clattr': clattr,
    #                       'index': index, 'stacks': stacks}
    #             e.submit(doFilter, **kwargs)
    #         else:
    #             e.submit(doNothing)

    return stacks

# def doNothing():
#     while True:
#         print 'nothing'
#         time.sleep(1)

def run_MedianFilter(image, device=None):
    """
    It works!

    :param image:
    :param device:
    :return:
    """

    pipeline = [mf.MedianFilter()]
    stacks = runPipeline(image, pipeline, device=device)
    return reconstruct_final_image(stacks)


def run_FFTFilter(image, FFTChoice='Forward', device=None):

    """
    It works!

    :param image:
    :param FFTChoice:
    :param device:
    :return:
    """

    pipeline = [fft.FFTFilter(FFTChoice=FFTChoice)]
    return runPipeline(image, pipeline, device=device)

def run_BilateralFilter(image, spatialRadius=3, rangeRadius=30, device=None):

    """
    It works!

    :param image:
    :param spatialRadius:
    :param rangeRadius:
    :param device:
    :return:
    """

    pipeline = [bf.BilateralFilter(spatialRadius=spatialRadius, rangeRadius=rangeRadius)]
    return runPipeline(image, pipeline, device=device)

def run_MaskFilter(image, maskChoice='mask3D', mask='StructuredElementL', L=3, device=None):
    """
    NOT WORKING

    """
    pipeline = [mskf.MaskFilter(maskChoice=maskChoice, mask=mask, L=L)]
    return runPipeline(image, pipeline, device=device)

def run_MMFilterDil(image, mask='StructuredElementL', L=3, device=None):

    """
    It works!

    :param image:
    :param mask:
    :param L:
    :param device:
    :return:
    """
    pipeline = [mmdil.MMFilterDil(mask=mask,L=L)]
    return run_f3d(image, pipeline, device=device)

def run_MMFilterEro(image, mask="StructuredElementL", L=3, device=None):
    """
    It works!

    :param image:
    :param mask:
    :param L:
    :param device:
    :return:
    """
    pipeline = [mmero.MMFilterEro(mask=mask, L=L)]
    return runPipeline(image, pipeline, device=device)

def run_MMFilterClo(image, mask='StructuredElementL', L=3, device=None):

    """
    It works!

    :param image:
    :param mask:
    :param L:
    :param device:
    :return:
    """

    pipeline = [mmclo.MMFilterClo(mask=mask, L=L)]
    return runPipeline(image, pipeline, device=device)

def run_MMFilterOpe(image, mask='StructuredElementL', L=3, device=None):

    pipeline = [mmope.MMFilterOpe(mask=mask, L=L)]
    stacks = runPipeline(image, pipeline, device=device)
    return reconstruct_final_image(stacks)

def reconstruct_final_image(stacks):

        stacks = sorted(stacks)
        image = stacks[0].stack

        # for stack in stacks:
            # print stack.endRange - stack.startRange

        for stack in stacks[1:]:
            image = np.append(image, stack.stack, axis=0)

        return image

def doFilter(image, pipeline, attr, clattr, index, stacks):

    global startIndex

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
            filter = pipeline[i]
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
            filter.releaseKernel()
        result = clattr.writeNextData(attr, stackRange[0], stackRange[1], maxOverlap)
        addResultStack(stacks, stackRange[0], stackRange[1], result, clattr.device.name, pipelineTime)

    clattr.inputBuffer.release()
    clattr.outputBuffer.release()
    if clattr.outputTmpBuffer is not None:
        clattr.outputTmpBuffer.release()

    startIndex = 0
    return stacks

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
        sr = helpers.StackRange()
        sr.startRange = startRange
        sr.endRange = endRange
        sr.stack = output
        sr.time = pipelineTime
        sr.name = name

        stacks.append(sr)

def setup_cl_prereqs(device=None):
    try:
        if not device:
            context, device, queue = ClAttributes.create_cl_attributes()
        if type(device) is not list:
            device = [device]
            context = [cl.Context(device)]
            queue = [cl.CommandQueue(context[0], device[0])]
        else:
            context = []; queue = []
            for i in range(len(device)):
                context.append(cl.Context([device[i]]))
                queue.append(cl.CommandQueue(context[i], device[i]))
    except TypeError:
        raise TypeError("\'device\' argument must be of type pyopencl.Device")

    return device, context, queue


# tests

