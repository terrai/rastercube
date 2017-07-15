"""
Global Land Cover Facility GLCF MCD12Q1
http://glcf.umd.edu/data/lc/
"""
import numpy as np
import numpy.ma as ma
import pylab as pl
import matplotlib.patches as mpatches

CLASSES_NAMES = {
    0: 'Water',
    1: 'Evergreen needleleaf forest',
    2: 'Evergreen broadleaf forest',
    3: 'Deciduous needleleaf forest',
    4: 'Deciduous broadleaf forest',
    5: 'Mixed forest',
    6: 'Closed shrublands',
    7: 'Open shrublands',
    8: 'Woody savannas',
    9: 'Savannas',
    10: 'Grasslands',
    11: 'Permanent wetlands',
    12: 'Croplands',
    13: 'Urban and built-up',
    14: 'Cropland/Natural vegetation mosaic',
    15: 'Snow and ice',
    16: 'Barren or sparsely vegetated',
    254: 'Unclassified',
    255: 'Fill value',
}

CMAP = {
    0: (31, 120, 180),
    1: (51, 160, 44),
    2: (51, 121, 44),
    3: (178, 223, 138),
    4: (178, 188, 138),
    5: (90, 160, 44),
    6: (119, 160, 44),
    7: (104, 160, 44),
    8: (205, 191, 111),
    9: (202, 160, 44),
    10: (51, 219, 44),
    11: (166, 206, 227),
    12: (255, 127, 0),
    13: (106, 106, 106),
    14: (255, 77, 0),
    15: (36, 243, 253),
    16: (220, 240, 0),
    254: (255, 0, 255),
    255: (255, 0, 255),
}


def glcf_to_rgb(arr):
    arr_rgb = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    for glcf_type, color in CMAP.items():
        arr_rgb[arr == glcf_type] = color
    return arr_rgb


def plot_glcf_labelmap(labels, ax=None):
    if ax is None:
        ax = pl.subplot(111)

    vimg = glcf_to_rgb(labels)
    vimg[labels.mask] = (0, 0, 0)
    ax.imshow(vimg, interpolation='nearest')

    lgd_patches = []
    for glcf_type in sorted(np.unique(labels)):
        if glcf_type is ma.masked:
            continue
        lgd_patches.append(
            mpatches.Patch(
                color=np.array(CMAP[glcf_type]) / 255.,
                label=CLASSES_NAMES[glcf_type]
            )
        )
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              handles=lgd_patches)
