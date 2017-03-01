# import POCLFilter
import pyopencl as cl
import time

import concurrent.futures as cf

import ClAttributes
import FilterAttributes
import filters.MedianFilter as mf
import filters.FFTFilter as fft
import helpers

def runPipeline(image, pipeline, device=None):

    startIndex = 0
    device, context, queue = setup_cl_prereqs(device=device)
    stacks = []

    # at this point we have a list of devices, contexts, and queues
    with cf.ThreadPoolExecutor(len(device)) as e:
        for i in range(len(device)):
            dev = device[i]
            ctx = context[i]
            q = queue[i]
            overlapAmount = 0
            atts = FilterAttributes.FilteringAttributes()
            index = i #what is this used for?
            clattr = ClAttributes.ClAttributes(ctx, dev, q, None, None, None)
            clattr.setMaxSliceCount(image)
            kwargs = {'image': image, 'pipeline': pipeline, 'overlapAmount': overlapAmount, 'attr': atts,
                      'startIndex': startIndex, 'clattr': clattr, 'index': index, 'stacks': stacks}
            e.submit(doFilter, **kwargs)

    return stacks

def run_MedianFilter(image, device=None):

    pipeline = [mf.MedianFilter()]
    return runPipeline(image, pipeline, device=device)

def run_FFTFilter(image, FFTChoice='Forward', device=None):

    pipeline = [fft.FFTFilter(FFTChoice=FFTChoice)]
    return runPipeline(image, pipeline, device=device)


def doFilter(image, pipeline, overlapAmount, attr, startIndex, clattr, index, stacks):

    start = startIndex
    maxOverlap = 0
    for filter in pipeline:
        maxOverlap = max(maxOverlap, filter.getInfo().overlapZ)

    maxSliceCount = clattr.maxSliceCount
    clattr.initializeData(image, attr, overlapAmount, maxSliceCount)

    for filter in pipeline:
        if filter.getInfo().useTempBuffer:
            clattr.outputTmpBuffer = cl.Buffer(clattr.context, cl.mem_flags.READ_WRITE, clattr.inputBuffer.size)
            break

    stackRange = [0, 0]
    while True:
        if start >= image.shape[0]:
            break
        start = getNextRange(image, start, stackRange, maxSliceCount)
        attr.sliceStart = stackRange[0]
        attr.sliceEnd = stackRange[1]
        clattr.loadNextData(image, attr, stackRange[0], stackRange[1], maxOverlap)
        maxSliceCount = stackRange[1] - stackRange[0]

        pipelineTime = 0
        for i in range(len(pipeline)):
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

    return stacks

def addResultStack(stacks, startRange, endRange, output, name, pipelineTime):

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

def getNextRange(image, startIndex, range, sliceCount):
    endIndex = image.shape[0]
    range[0] = startIndex
    range[1] = startIndex + sliceCount
    if range[1] >= endIndex:
        range[1] = endIndex
    startIndex = range[1]
    return startIndex


def test_median():
    import tifffile
    image = tifffile.imread('/media/winHDD/hparks/rec20160525_165348_holland_polar_bear_hair_1.tif')
    image = helpers.scale_to_uint8(image)[:10]

    stacks = run_MedianFilter(image)
    # for stack in stacks:
    #     print stack.stack
    tifffile.imsave('/home/hparks/Desktop/test2.tif', stacks[0].stack)

def test_fft():
    import tifffile
    image = tifffile.imread('/media/winHDD/hparks/rec20160525_165348_holland_polar_bear_hair_1.tif')
    image = helpers.scale_to_uint8(image)[:10]

    # stacks = run_FFTFilter(image)
    stacks = run_FFTFilter(image, 'Inverse')
    # for stack in stacks:
    #     print stack.stack
    # tifffile.imsave('/home/hparks/Desktop/forward.tif', stacks[0].stack)
    tifffile.imsave('/home/hparks/Desktop/inverse.tif', stacks[0].stack)

if __name__ == '__main__':
    test_fft()
