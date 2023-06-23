from typing import Tuple

import numpy as np
from numpy.linalg import solve


def gmt_coords(
        rows:int, cols:int, gmt: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
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
    # coord = (x, y, 1) Homogeneous coordinates
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


def gmt_nearest_neighbor_interpolation(
    image: np.ndarray,
    coords: np.ndarray,
    shape: Tuple[int, int]
) -> np.ndarray:
    """Apply a nearest neighbor interpolation to a pixel of an image.
    
    Parameters:
        image: input image
        coords: coordinates of the pixel
        shape: shape of the output image
    Returns:
        image_nnb: interpolated image
    """

    # Get the coordinates of the pixels
    rows, cols = image.shape
    x_o, y_o = coords
    x_o = np.clip(x_o, 0, cols - 1)
    y_o = np.clip(y_o, 0, rows - 1)

    # Get the coordinates of the neighbor
    x_n = np.round(x_o).astype(np.uint32)
    y_n = np.round(y_o).astype(np.uint32)

    image_blnr = image[y_n, x_n]
    return image_blnr.reshape(shape).astype(image.dtype)

def gmt_bilinear_interpolation(
    image: np.ndarray,
    coords: np.ndarray,
    shape: Tuple[int, int]
) -> np.ndarray:
    """Apply a bilinear interpolation to a pixel of an image.
    
    Parameters:
        image: input image
        coords: coordinates of the pixel
        shape: shape of the output image
    Returns:
        image_blnr: interpolated image
    """

    # Get the coordinates of the pixels
    rows, cols = image.shape
    x_o, y_o = coords
    x_o = np.clip(x_o, 0, cols - 1)
    y_o = np.clip(y_o, 0, rows - 1)

    # Get the coordinates of the neighbor
    x_n = np.floor(x_o).astype(np.uint32)
    y_n = np.floor(y_o).astype(np.uint32)


    # Get the distances
    dx = x_o - x_n
    dy = y_o - y_n

    # Get the weights of the pixels
    weights = np.array([
        (1 - dx)*(1 - dy),
        dx*(1 - dy),
        (1 - dx)*dy,
        dx*dy
    ])

    # Ensure that the coordinates are inside the image
    xn_1 = np.where(x_n + 1 < cols - 1, x_n + 1, cols - 1)
    yn_1 = np.where(y_n + 1 < rows - 1, y_n + 1, rows - 1)

    # Get the neighborhood of the pixel
    neighborhood = np.vstack((
        image[y_n, x_n],
        image[y_n, xn_1],
        image[yn_1, x_n],
        image[yn_1, xn_1]
    ))

    image_blnr = np.sum(weights * neighborhood, axis=0)
    return image_blnr.reshape(shape).astype(image.dtype)

def gmt_bilinear_interpolation_naive(
        image: np.ndarray, x_o: float, y_o: float
) -> float:
    """Apply a bilinear interpolation to a pixel of an image.
    
    Parameters:
        image: input image
        x_o: x coordinate of the pixel
        y_o: y coordinate of the pixel
    Returns:
        pixel: interpolated pixel
    """
    
    # Get the coordinates of the pixels
    rows, cols = image.shape
    x_o = np.clip(x_o, 0, cols - 1)
    y_o = np.clip(y_o, 0, rows - 1)
    x_i, y_i = int(x_o), int(y_o)

    # Get the neighborhood of the pixel
    p1 = image[y_i, x_i]
    p2 = image[y_i, x_i + 1]
    p3 = image[y_i + 1, x_i]
    p4 = image[y_i + 1, x_i + 1]

    # Get the distances
    dx = x_o - x_i
    dy = y_o - y_i

    # Get the weights of the pixels
    weights = np.array([
        (1 - dx)*(1 - dy),
        dx*(1 - dy),
        (1 - dx)*dy,
        dx*dy
    ])

    # Calculate the pixel intensity
    neighbors = np.array([p1, p2, p3, p4])
    pixel = weights @ neighbors.T
    return pixel


def gmt_bicubic_interpolation(
        image: np.ndarray, coords: np.ndarray, shape: np.ndarray
) -> np.ndarray:
    """Compute bicubic interpolation using B-splines

    Params:
        image: Input image
        coords: Coordinates of output image
        shape: Shape of output image
    Returns:
        out: Output image
    """

    def R(s: np.ndarray) -> np.ndarray:
        """ R Cubic B-spline function
        
        Params:
            s: Input array
        Returns:
            R: R function values
        """

        R = (
            np.power(
                np.maximum(s+2, 0), 3
            ) -
            4 * np.power(
                np.maximum(s+1, 0), 3
            ) +
            6 * np.power(
                np.maximum(s, 0), 3
            ) -
            4 * np.power(
                np.maximum(s-1, 0), 3
            )
        ) / 6

        return R

    # Compute the indices of the pixels
    rows, cols = image.shape
    x, y = coords
    x = np.clip(x, 0, cols - 1)
    y = np.clip(y, 0, rows - 1)
    x0 = np.floor(x).astype(np.uint32)
    y0 = np.floor(y).astype(np.uint32)

    # Compute the fractional part of the coordinates
    dx = x - x0
    dy = y - y0

    # Compute interpolation weights
    weights = np.array(
        [R(m-dx) * R(n-dy) for m in range(-1, 3) for n in range(-1, 3)]
    )

    # Pad the image
    padded_image = np.pad(image, ((2, 2), (2, 2)), mode='constant')

    # Compute the interpolated pixels
    pixels = np.array(
        [padded_image[y0+2+j, x0+2+i] for j in range(-1, 3) for i in range(-1, 3)]
    )

    # Compute the output image
    out = np.sum(weights * pixels, axis=0).reshape(shape)
    return out


def gmt_lagrange_interpolation(
        image: np.ndarray, coords: np.ndarray, shape: np.ndarray
) -> np.ndarray:
    """Compute lagrange interpolation

    Params:
        image: Input image
        coords: Coordinates of output image
        shape: Shape of output image
    Returns:
        out: Output image
    """
    
    # Compute the indices of the pixels
    rows, cols = image.shape
    x, y = coords
    x = np.clip(x, 0, cols - 1)
    y = np.clip(y, 0, rows - 1)
    x0 = np.floor(x).astype(np.uint32)
    y0 = np.floor(y).astype(np.uint32)

    # Compute the fractional part of the coordinates
    dx = x - x0
    dy = y - y0

    # Compute L interpolation weights
    L_w = np.array(
        [
            -dx*(dx-1)*(dx-2)/6,
            (dx+1)*(dx-1)*(dx-2)/2,
            -dx*(dx+1)*(dx-2)/2,
            dx*(dx+1)*(dx-1)/6
        ]
    )

    # Compute interpolation weights
    w = np.array(
        [
            -dy*(dy-1)*(dy-2)/6,
            (dy+1)*(dy-1)*(dy-2)/2,
            -dy*(dy+1)*(dy-2)/2,
            dy*(dy+1)*(dy-1)/6
        ]
    )

    # Pad the image
    padded_image = np.pad(image, ((2, 2), (2, 2)), mode='constant')

    # Compute the interpolated pixels
    pixels = np.array(
        [padded_image[y0+n, x0+2+i] for n in range(1, 5) for i in range(-1, 3)]
    )

    # Compute the Lagrange interpolation
    L = np.array(
        [np.sum(L_w * pixels[n*4:(n+1)*4], axis=0) for n in range(4)]
    )

    # Compute the output image
    out = np.sum(w * L, axis=0).reshape(shape)

    return out


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