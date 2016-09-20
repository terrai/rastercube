# %%cython --annotate --compile-args=-fopenmp --link-args=-fopenmp --force
"""
Cython implementation of LANDSAT 8 qa interpretation
http://landsat.usgs.gov/qualityband.php
"""
from cython.view cimport array as cvarray
from cython.parallel import parallel, prange, threadid
from cython.view cimport array as cvarray
from cython cimport view

import numpy as np
cimport cython
cimport numpy as np
cimport openmp

from libc.math cimport fabs, log10, floor, exp, log, M_PI
from libc.stdio cimport printf
from libc.float cimport DBL_MIN

@cython.cdivision(True)
@cython.profile(False)
@cython.boundscheck(False)
cdef np.float32_t landsat8_pix_qa_to_qaconf(np.uint16_t lanqa) nogil:
    """
    Transforms LANDSAt QA into a confidence score.
    confidence comes from the vi_usefulness bits
    """
    # http://landsat.usgs.gov/qualityband.php

    # 0 - Designated Fill
    if lanqa & 0x1 == 1:
        return 0

    # 1 - Dropped frame
    if (lanqa >> 1) & 0x1 == 1:
        return 0

    # 2 - Terrain occlusion
    if (lanqa >> 2) & 0x1 == 1:
        return 0

    # 3 - Reserved

    # 4-5 Water Confidence
    if (lanqa >> 4) & 0x3 == 3:
        return 0

    # 6-7 Reserved

    # 8-9 Vegetation confidence
    # ignore as if often seems to be undetermined
    #if (lanqa >> 8) & 0x3 != 3:
    #    return 0

    # 10-11 Snow confidence
    if (lanqa >> 10) & 0x3 == 3:
        return 0

    # 12-13 Cirrus confidence
    if (lanqa >> 12) & 0x3 == 3:
        return 0

    # 14-15 Cloud confidence
    if (lanqa >> 14) & 0x3 == 3:
        return 0

    return 1

@cython.linetrace(False)
@cython.boundscheck(False)
def landsat8_qa_to_qaconf(qa):
    qaconf = np.zeros_like(qa, dtype=np.float32)

    cdef np.uint16_t[:] c_qa=qa.reshape(-1)
    cdef np.float32_t[:] c_qaconf=qaconf.reshape(-1)

    cdef np.int64_t n = np.prod(qa.shape)
    cdef np.int64_t i = 0

    with nogil, parallel():
        for i in prange(n, schedule='guided'):
            c_qaconf[i] = landsat8_pix_qa_to_qaconf(c_qa[i])

    return qaconf
