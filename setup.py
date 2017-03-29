from setuptools import setup, find_packages

setup(
    name='pyF3D',
    version='1.0.0',
    description='Filtering for micro-tomography data',
    long_description='Read from README file',
    url='https://github.com/holdymoldy/pyF3D',
    author='Holden Parks',
    author_email='hparks@lbl.gov',
    packages=find_packages(),
    license='Something',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: Something',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7'
    ],
    install_requires=['numpy', 'pyopencl']
)