from os import path
from setuptools import setup, find_packages
from codecs import open


here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pyf3d',
    version='1.0.3',
    description='Filtering for micro-tomography data',
    long_description=long_description,
    url='https://github.com/holdymoldy/pyF3D',
    author='Holden Parks',
    author_email='hparks@lbl.gov',
    packages=find_packages(),
    package_data={'pyF3D':['OpenCL/*']},
    license='BSD',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: BSD License',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7'
	'Programming Language :: Python :: 3.4'
    ],

    install_requires=['pyopencl', 'numpy', 'futures', 'tifffile']
)
