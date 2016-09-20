"""
To rebuild cython, run with

    python setup.py build_ext

"""
import os
import numpy
import platform
import glob
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

IS_OSX = platform.system() == 'Darwin'

if IS_OSX:
    # Build with gcc because we use libstdc++ (that's what conda is built
    # with) and osx' xcode ships with a pre c++11 libstdc++ (which causes
    # errors because the c++11 stuff is in tr1/).
    # Also, clang on osx doesn't support openmp
    os.environ['CXX'] = '/usr/local/bin/g++-4.9'
    os.environ['CC'] = '/usr/local/bin/gcc-4.9'

    cflags = []
    ldflags = []
else:
    # Note that if you run on a spark cluster, you have to compile the
    # code on each node in the cluster, otherwise if the nodes have different
    # architectures, you might run into crashes
    cflags = ['-march=native']
    ldflags = []

cflags += ['-std=c++11', '-O3', '-fopenmp']
ldflags += ['-std=c++11', '-fopenmp']

landsat8qa = Extension(
    'rastercube.datasources.landsat8_qa',
    ['rastercube/datasources/landsat8_qa.pyx'],
    include_dirs=[numpy.get_include()],
    extra_compile_args=cflags,
    extra_link_args=ldflags,
)

modqa = Extension(
    'rastercube.datasources.modis_qa',
    ['rastercube/datasources/modis_qa.pyx'],
    include_dirs=[numpy.get_include()],
    extra_compile_args=cflags,
    extra_link_args=ldflags,
)


setup(
    name='rastercube',
    version='0.1',
    # So that it finds rastercube.<subpackage>
    packages=find_packages(include=["rastercube*"]),
    ext_modules=cythonize([landsat8qa, modqa]),
    install_requires=[
        'numpy',
        'scipy',
        'sklearn',
        'requests',
        'joblib',
        'pyprind',
        'hdfs',
        'coverage',
        'beautifulsoup4',
        'joblib',
        'PIL',
        'nbconvert==4.2.0',
    ],
    setup_requires=[
        'pytest-runner',
        'cython',
        'flake8',
    ],
    test_requires=[
        'pytest',
        'pytest-cov',
    ],
    package_data={
        'rastercube': ['assets/*'],
    },
    zip_safe=True,
)
