# %%cython --annotate --compile-args=-fopenmp --link-args=-fopenmp --force
"""
Cython implementation of MODIS qa interpretation, which takes a long time
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
cdef np.float32_t pix_qa_to_qaconf(np.uint16_t modqa) nogil:
    """
    Transforms MODIS QA into a confidence score.
    confidence comes from the vi_usefulness bits
    """
    # https://lpdaac.usgs.gov/dataset_discovery/modis/modis_products_table/mod13q1

    # 0-1 MODLAND_QA
    #   00    VI produced, good quality
    #   01    VI produced, but check other QA
    #   10    Pixel produced, but most probably cloudy
    #   11    Pixel not produced due to other reasons than clouds
    # We discards 11 and check quality for 00, 01 and 10
    # TODO: Should we discard 10 to ?
    if modqa & 0x3 == 3:
        return 0

    # 6-7 Aerosol quantity
    #  00    Climatology
    #  01    Low
    #  10    Average
    #  11    High
    if (modqa >> 6) & 0x3 == 3:
        return 0

    # 8 Adjacent cloud detected
    #  1    Yes
    #  0    No
    if (modqa >> 8) & 0x1 == 1:
        return 0

    # 9 Atmosphere BRDF correction performed
    #  1    Yes
    #  0    No
    # TODO: Not sure, but I think we can ignore this. Terra-i does ignore it
    #atm_corr = (qa_data >> 9) & 0x1
    #valid &= atm_corr == 0

    # 10 Mixed Clouds
    #  1    Yes
    #  0    No
    if (modqa >> 10) & 0x1 == 1:
        return 0

    # 11-13 Land/Water Flag
    #  000    Shallow ocean
    #  001    Land (Nothing else but land)
    #  010    Ocean coastlines and lake shorelines
    #  011    Shallow inland water
    #  100    Ephemeral water
    #  101    Deep inland water
    #  110    Moderate or continental ocean
    #  111    Deep ocean
    # We discard everything but land
    if (modqa >> 11) & 0x7 != 1:
        return 0

    # 14 Possible snow/ice
    #  1    Yes
    #  0    No
    if (modqa >> 14) & 0x1 == 1:
        return 0

    # 15 Possible shadow
    #  1    Yes
    #  0    No
    if (modqa >> 15) & 0x1 == 1:
        return 0

    # 2-5 VI usefulness
    #  0000    Highest quality
    #  0001    Lower quality
    #  0010    Decreasing quality
    #  0100    Decreasing quality
    #  1000    Decreasing quality
    #  1001    Decreasing quality
    #  1010    Decreasing quality
    #  1100    Lowest quality
    #  1101    Quality so low that it is not useful
    #  1110    L1B data faulty
    #  1111    Not useful for any other reason/not processed
    return 1. - <np.float32_t>((modqa >> 2) & 0xf) / 12.

@cython.linetrace(False)
@cython.boundscheck(False)
def qa_to_qaconf(qa):
    qaconf = np.zeros_like(qa, dtype=np.float32)

    cdef np.uint16_t[:] c_qa=qa.reshape(-1)
    cdef np.float32_t[:] c_qaconf=qaconf.reshape(-1)

    cdef np.int64_t n = np.prod(qa.shape)
    cdef np.int64_t i = 0

    with nogil, parallel():
        for i in prange(n, schedule='guided'):
            c_qaconf[i] = pix_qa_to_qaconf(c_qa[i])

    return qaconf
