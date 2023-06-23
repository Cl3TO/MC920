from pathlib import Path

from skimage import transform, io

from gmt.gmt import gmt_rotate, gmt_scale, gmt_resize


def test_gmt_rotate():
    """
    Testa a função gmt_rotate.
    """
    image = io.imread(Path('../gmt/images/baboon.png'))

    image_rotated = gmt_rotate(image, 45)

    

def test_gmt_scale():
    """
    Testa a função gmt_scale.
    """
    ...

def test_gmt_resize():
    """
    Testa a função gmt_resize.
    """
    ...