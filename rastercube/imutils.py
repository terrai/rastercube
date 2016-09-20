import numpy as np
from PIL import Image, ImageDraw


def rasterize_poly(poly_xy, shape):
    """
    Args:
        poly_xy: [(x1, y1), (x2, y2), ...]
    Returns a bool array containing True for pixels inside the polygon
    """
    _poly = poly_xy[:-1]
    # PIL wants *EXACTLY* a list of tuple (NOT a numpy array)
    _poly = [tuple(p) for p in _poly]

    img = Image.new('L', (shape[1], shape[0]), 0)
    ImageDraw.Draw(img).polygon(_poly, outline=0, fill=1)
    return np.array(img) == 1
