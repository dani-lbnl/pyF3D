from FilterManager import *

def test_median():
    image = tifffile.imread('/media/winHDD/hparks/rec20160525_165348_holland_polar_bear_hair_1.tif')
    image = scale_to_uint8(image)
    platforms = ClAttributes.list_all_cl_platforms()
    d = {}
    for p in platforms:
        d[p] = 100

    image = run_MedianFilter(image, d)
    tifffile.imsave('/home/hparks/Desktop/median.tif', image)

def test_fft():
    image = tifffile.imread('/media/winHDD/hparks/rec20160525_165348_holland_polar_bear_hair_1.tif')
    image = scale_to_uint8(image)[:10]

    im = run_FFTFilter(image, 'Inverse')
    tifffile.imsave('/home/hparks/Desktop/inverse.tif', im)

def test_bilateral():
    image = tifffile.imread('/media/winHDD/hparks/rec20160525_165348_holland_polar_bear_hair_1.tif')
    image = scale_to_uint8(image)[:10]

    im = run_BilateralFilter(image, 3, 30)
    tifffile.imsave('/home/hparks/Desktop/bilateral.tif', im)

def test_mask():
    image = tifffile.imread('/media/winHDD/hparks/rec20160525_165348_holland_polar_bear_hair_1.tif')
    image = scale_to_uint8(image)[:100]

    mask = np.zeros((100,352,275))

    im = run_MaskFilter(image, mask=mask)
    tifffile.imsave('/home/hparks/Desktop/maskfilter.tif', im)

def test_mmdil():
    image = tifffile.imread('/media/winHDD/hparks/rec20160525_165348_holland_polar_bear_hair_1.tif')
    image = scale_to_uint8(image)


    im = run_MMFilterDil(image, mask='StructuredElementL')
    tifffile.imsave('/home/hparks/Desktop/mmdil.tif', im)

def test_mmero():
    image = tifffile.imread('/media/winHDD/hparks/rec20160525_165348_holland_polar_bear_hair_1.tif')
    image = scale_to_uint8(image)[:10]

    im = run_MMFilterEro(image, mask='Diagonal10x10x10')
    tifffile.imsave('/home/hparks/Desktop/mmero.tif',im)

def test_mmclo():
    image = tifffile.imread('/media/winHDD/hparks/rec20160525_165348_holland_polar_bear_hair_1.tif')
    image = scale_to_uint8(image)[:10]

    im = run_MMFilterClo(image, mask='StructuredElementL')
    tifffile.imsave('/home/hparks/Desktop/mmclo.tif', im)

def test_mmope():
    import time
    image = tifffile.imread('/media/winHDD/hparks/rec20160525_165348_holland_polar_bear_hair_1.tif')
    image = scale_to_uint8(image)

    im = run_MMFilterOpe(image, mask='Diagonal10x10x4')
    tifffile.imsave('/home/hparks/Desktop/mmope.tif', im)

def test_pipeline():

    image = tifffile.imread('/media/winHDD/hparks/rec20160525_165348_holland_polar_bear_hair_1.tif')
    image = scale_to_uint8(image)

    pipeline = [mf.MedianFilter(), mmdil.MMFilterDil(), bf.BilateralFilter(), mf.MedianFilter()]
    platforms = ClAttributes.list_all_cl_platforms()
    # im = run_f3d(image, pipeline, platforms)
    im = run_f3d(image, pipeline)

    tifffile.imsave('/home/hparks/Desktop/pipeline.tif', im)


if __name__ == '__main__':
    import tifffile

    import os
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

    test_median()
    # test_pipeline()
    # test_mask()
