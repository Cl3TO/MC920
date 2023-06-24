from pathlib import Path
from skimage import io
import numpy as np
from numpy.linalg import solve
from matplotlib import pyplot as plt
from typing import Tuple

def gmt_bilinear_interpolation_vet(
    image: np.ndarray, coords: np.ndarray, shape: Tuple[int, int]
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
    weights = np.array(
        [(1 - dx) * (1 - dy), dx * (1 - dy), (1 - dx) * dy, dx * dy]
    )

    xn_1 = np.where(x_n + 1 < cols - 1, x_n + 1, cols - 1)
    yn_1 = np.where(y_n + 1 < rows - 1, y_n + 1, rows - 1)

    # Get the neighborhood of the pixel
    neighborhood = np.vstack(
        (
            image[y_n, x_n],
            image[y_n, xn_1],
            image[yn_1, x_n],
            image[yn_1, xn_1],
        )
    )

    image_blnr = np.sum(weights * neighborhood, axis=0)
    return image_blnr.reshape(shape).astype(image.dtype)


def gmt_map_vet(
    image: np.ndarray, gmt: np.ndarray, inverse: bool = False
) -> np.ndarray:
    """Apply a geometric transformation to an image.
    (Vectorized version)
    Parameters:
        image: input image
        gmt: transformation matrix (3x3)
        inverse: if True, apply the inverse transformation
    Returns:
        t_image: transformed image
    """

    # Get the coordinates of the pixels
    rows, cols = image.shape
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
    t_coords = t_coords[:2] / t_coords[2]
    # t_coords = np.round(t_coords[:2] / t_coords[2]).astype(np.int32)

    # Create the transformed image
    t_image = np.zeros_like(image)
    # Check if the pixel is inside the image
    inside = (
        (t_coords[0] >= 0)
        & (t_coords[1] >= 0)
        & (t_coords[0] < cols)
        & (t_coords[1] < rows)
    )
    if inverse:
        t_image = gmt_bilinear_interpolation_vet(image, t_coords, image.shape)
    else:
        t_coords = np.round(t_coords).astype(np.int32)
        t_image[t_coords[1][inside], t_coords[0][inside]] = image[
            coords[1][inside], coords[0][inside]
        ]

    return t_image

if __name__ == "__main__":
    """Script que aplica uma transformação perspectiva em uma imagem."""
    # Path to the images
    images_path = Path('images')

    # Read the images
    image = io.imread(images_path / 'baboon_perspectiva.png')
    image = image[:, :, 0]

    # Define os pontos de origem e destino
    src = np.array([[0, 0], [511, 0], [511, 511], [0, 511]])
    dst = np.array([[37, 51], [342, 42], [485, 467], [73, 380]])

    # Cria a matriz de coeficientes do sistema de equações
    A = np.zeros((8, 8))
    for i in range(4):
        A[i * 2, :] = [
            src[i, 0],
            src[i, 1],
            1,
            0,
            0,
            0,
            -dst[i, 0] * src[i, 0],
            -dst[i, 0] * src[i, 1],
        ]
        A[i * 2 + 1, :] = [
            0,
            0,
            0,
            src[i, 0],
            src[i, 1],
            1,
            -dst[i, 1] * src[i, 0],
            -dst[i, 1] * src[i, 1],
        ]

    # Cria o vetor de soluções do sistema de equações
    b = np.zeros((8,))
    for i in range(4):
        b[i * 2] = dst[i, 0]
        b[i * 2 + 1] = dst[i, 1]

    # Resolve o sistema de equações
    h = solve(A, b)

    # Cria a matriz de projeção perspectiva
    P = np.zeros((3, 3))
    P[0, :] = h[:3]
    P[1, :] = h[3:6]
    P[2, :2] = h[6:]
    P[2, 2] = 1

    # Aplica a transformação
    image_transformed = gmt_map_vet(image, P, inverse=True)
    
    # Salva a imagem
    plt.imshow(image_transformed, cmap='gray')
    plt.axis('off')
    plt.savefig('baboon_perspectiva_transformada.png', bbox_inches='tight', pad_inches=0)