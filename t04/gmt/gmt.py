from typing import Tuple

import numpy as np
from numpy.linalg import solve


def gmt_coords(rows:int, cols:int, gmt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get the coordinates to apply a geometric transformation to an image.

    Parameters:
        rows: number of rows of the image
        cols: number of columns of the image
        gmt: transformation matrix (3x3)
    Returns:
        tuple:
            coords: coordinates of the transformed pixels
            t_coords: transformed coordinates
    """

    # Get the coordinates of the pixels
    indeces = np.indices((rows, cols))
    # coord = (x, y, 1)
    coords = np.vstack(
        (
            indeces[1].ravel(),
            indeces[0].ravel(),
            np.ones(rows * cols, dtype=np.int32),
        )
    )

    # Apply the transformation
    t_coords = gmt @ coords

    # Normalize the coordinates
    t_coords = (t_coords[:2] / t_coords[2])

    # Return the coordinates of the pixels (x, y)
    return coords[:2], t_coords


def gmt_warp(
    image: np.ndarray, gmt: np.ndarray, inverse: bool = False
) -> np.ndarray:
    """Apply a geometric transformation to an image.

    Parameters:
        image: input image
        gmt: transformation matrix (3x3)
        inverse: if True, apply the inverse transformation
    Returns:
        t_image: transformed image
    """


    # Create the transformed image
    t_image = np.zeros_like(image)


    # get rows and cols
    rows, cols = image.shape

    # Get the coordinates of the pixels
    coords, t_coords = gmt_coords(rows, cols, gmt)

    # Round the coordinates
    t_coords = np.round(t_coords).astype(np.int32)

    # Check if the coordinates are inside the image
    inside = np.logical_and(
        np.logical_and(t_coords[0] >= 0, t_coords[0] < cols),
        np.logical_and(t_coords[1] >= 0, t_coords[1] < rows),
    )

    if inverse:
        t_image[coords[1][inside], coords[0][inside]] = (
            image[t_coords[1][inside], t_coords[0][inside]]
        )
    else:
        t_image[t_coords[1][inside], t_coords[0][inside]] = (
            image[coords[1][inside], coords[0][inside]]
        )

    return t_image