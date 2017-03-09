from FilterManager import *

def test_median():
    image = tifffile.imread('/media/winHDD/hparks/rec20160525_165348_holland_polar_bear_hair_1.tif')
    image = helpers.scale_to_uint8(image)
    devices = ClAttributes.list_all_cl_devices()


    image = run_MedianFilter(image, device=devices)
    tifffile.imsave('/home/hparks/Desktop/median.tif', image)

def test_fft():
    image = tifffile.imread('/media/winHDD/hparks/rec20160525_165348_holland_polar_bear_hair_1.tif')
    image = helpers.scale_to_uint8(image)[:10]

    # stacks = run_FFTFilter(image)
    stacks = run_FFTFilter(image, 'Inverse')
    # for stack in stacks:
    #     print stack.stack
    # tifffile.imsave('/home/hparks/Desktop/forward.tif', stacks[0].stack)
    tifffile.imsave('/home/hparks/Desktop/inverse.tif', stacks[0].stack)

def test_bilateral():
    image = tifffile.imread('/media/winHDD/hparks/rec20160525_165348_holland_polar_bear_hair_1.tif')
    image = helpers.scale_to_uint8(image)[:10]

    # stacks = run_FFTFilter(image)
    stacks = run_BilateralFilter(image, 3, 30)
    # for stack in stacks:
    #     print stack.stack
    tifffile.imsave('/home/hparks/Desktop/bilateral.tif', stacks[0].stack)

def test_mask():
    image = tifffile.imread('/media/winHDD/hparks/rec20160525_165348_holland_polar_bear_hair_1.tif')
    image = helpers.scale_to_uint8(image)[:10]

    stacks = run_MaskFilter(image, mask='Diagonal10x10x4')
    # for stack in stacks:
    #     print stack.stack
    tifffile.imsave('/home/hparks/Desktop/maskfilter.tif', stacks[0].stack)

def test_mmdil():
    image = tifffile.imread('/media/winHDD/hparks/rec20160525_165348_holland_polar_bear_hair_1.tif')
    image = helpers.scale_to_uint8(image)

    im = run_MMFilterDil(image, mask='StructuredElementL')
    # for stack in stacks:
    #     print stack.stack
    tifffile.imsave('/home/hparks/Desktop/mmdil.tif', im)

def test_mmero():
    image = tifffile.imread('/media/winHDD/hparks/rec20160525_165348_holland_polar_bear_hair_1.tif')
    image = helpers.scale_to_uint8(image)[:10]

    stacks = run_MMFilterEro(image, mask='Diagonal10x10x10')
    # for stack in stacks:
    #     print stack.stack
    tifffile.imsave('/home/hparks/Desktop/mmero.tif', stacks[0].stack)

def test_mmclo():
    image = tifffile.imread('/media/winHDD/hparks/rec20160525_165348_holland_polar_bear_hair_1.tif')
    image = helpers.scale_to_uint8(image)[:10]

    stacks = run_MMFilterClo(image, mask='StructuredElementL')
    # for stack in stacks:
    #     print stack.stack
    tifffile.imsave('/home/hparks/Desktop/mmclo.tif', stacks[0].stack)

def test_mmope():
    import time
    image = tifffile.imread('/media/winHDD/hparks/rec20160525_165348_holland_polar_bear_hair_1.tif')
    # image = helpers.scale_to_uint8(image)[:20]
    image = helpers.scale_to_uint8(image)
    devices = ClAttributes.list_all_cl_devices()
    devices.reverse()
    # print devices

    # start_time = time.time()
    im = run_MMFilterOpe(image, mask='Diagonal10x10x4', device=devices)
    # print "with multiple devices: ", time.time() - start_time
    start_time = time.time()
    # im = run_MMFilterOpe(image, mask='Diagonal10x10x4', device=devices[1])
    # print "with GPU: ", time.time() - start_time
    tifffile.imsave('/home/hparks/Desktop/mmope.tif', im)

def test_pipeline():

    image = tifffile.imread('/media/winHDD/hparks/rec20160525_165348_holland_polar_bear_hair_1.tif')
    image = helpers.scale_to_uint8(image)

    pipeline = [mf.MedianFilter(), mmdil.MMFilterDil(), bf.BilateralFilter(), mf.MedianFilter()]
    im = run_f3d(image, pipeline)
    # pipeline1 = [mf.MedianFilter()]
    # pipeline2 = [mmdil.MMFilterDil()]
    # pipeline.reverse()
    # im = run_f3d(image, pipeline1)
    # im = run_f3d(im, pipeline2)

    tifffile.imsave('/home/hparks/Desktop/pipeline.tif', im)


if __name__ == '__main__':
    import tifffile

    import os
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

    test_median()
    # test_pipeline()