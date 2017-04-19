
pyF3D
=====

About F3D
---------

F3D is a python package designed for high-resolution 3D image with kernels written in OpenCL. F3D achieves
platform-portable parallelism on modern multi-core CPUs and many-core GPUs. The interface and mechanisms to access F3D
accelerated kernels are written in Python to be fully integrated with other Python packages. F3D delivers several key
image-processing algorithms necessary to remove artifacts from micro-tomography data. The algorithms consist of data
parallel aware filters that can efficiently utilize resources and can process data out of core and scale efficiently
across multiple accelerators. Optimized for data parallel filters, F3D streams data out of core to efficiently manage
resources, such as memory, over complex execution sequence of filters. This has greatly expedited several scientific
workflows dealing with high-resolution images. F3D preforms two main types of 3D image processing operations:
non-linear filtering, such as bilateral and median filtering, and morphological operators (MM) with varying 3D
structuring elements.

Installation
------------

Linux
+++++

You may install pyF3D with either conda or pip:

.. code-block:: bash

  conda install -c als832 pyf3d

Or:

.. code-block:: bash

  pip install pyF3D


Copyright Notice
----------------

F3D Image Processing and Analysis for Many- and Multi-core Platforms, Copyright (c) 2014, The Regents of the University
of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S.
Dept. of Energy).  All rights reserved.

If you have questions about your rights to use or distribute this software, please contact Berkeley Lab's Technology
Transfer Department at  TTD@lbl.gov.

NOTICE.  This software is owned by the U.S. Department of Energy.  As such, the U.S. Government has been granted for
itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to
reproduce, prepare derivative works, and perform publicly and display publicly.  Beginning five (5) years after the
date permission to assert copyright is obtained from the U.S. Department of Energy, and subject to any subsequent five
(5) year renewals, the U.S. Government is granted for itself and others acting on its behalf a paid-up, nonexclusive,
irrevocable, worldwide license in the Software to reproduce, prepare derivative works, distribute copies to the public,
perform publicly and display publicly, and to permit others to do so.

