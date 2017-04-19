F3D offers the following as usable filters:

*   Median Filter
*   Bilateral Filter
*   Mask Filter
*   MM Filter: Opening
*   MM Filter: Closing
*   MM Filter: Dilation
*   MM Filter: Erosion

A FFT filter will be implemented in the future.

To use the filters, you may either use built-in functions for single filters, or create a list of filters. For example:

.. code-block:: python

    import pyF3D as f
    import tifffile as t

    image = t.imread('.../example.tif')
    median = f.run_MedianFilter(image)

This code will run a median filter on the image ``example.tif``. To use multiple filters, create a list of filters while
filling out optional parameters for each filter:

.. code-block:: python

    pipeline = [f.MedianFilter(), f.MMFilterEro(mask='Diagonal3x3x3')]
    new_image = f.run_f3d(image, pipeline)

This code will run a median filter, then an erosion filter, on the image. You may also specify the platforms on which
to run the calculations. F3D provides a helper function ``pyF3D.list_all_cl_platforms``, which will specify all
platforms you can access for processing. By default, F3D will use the first device on this list. There are several
ways to specify the platforms to use:

1. Specify a single OpenCL platform in an F3D function's arguments:

   .. code-block:: python

       platform = f.list_all_cl_platforms()[0]
       new_image = f.run_f3d(image, pipline, platform=platform)

2. Specify a list of platforms, for processing in parallel:

   .. code-block:: python

       platforms = f.list_all_cl_platforms()
       new_image = f.run_f3d(image, pipline, platform=platforms)

3. Specify a dictionary of platform/int key-value pairs. The ``int`` value specifies the maximum number of slices to be
   loaded onto the corresponding platform.

   .. code-block:: python

       platforms = {}
       for p in f.list_all_cl_platforms():
           platforms[p] = 200
       new_image = f.run_f3d(image, pipline, platform=platforms)

   This code ensures that no more than ``200`` slices of the image will be loaded onto each platform at any time.