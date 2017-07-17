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


cflags = ['-march=native', '-std=c++11', '-O3', '-fopenmp']
ldflags = ['-std=c++11', '-fopenmp']

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
        'pillow',
        'nbconvert',
        'nbsphinx',
        'gdal',
        'matplotlib',

    ],
    setup_requires=[
        'pytest-runner',
        'cython',
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
