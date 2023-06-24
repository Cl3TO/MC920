from pathlib import Path
from typing import Tuple

import numpy as np
from rich import print
from skimage import io
from typer import Option, run


def gmt_coords(
    rows: int, cols: int, gmt: np.ndarray
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
    t_coords = t_coords[:2] / t_coords[2]

    # Return the coordinates of the pixels (x, y)
    return coords[:2], t_coords


def gmt_nearest_neighbor_interpolation(
    image: np.ndarray, coords: np.ndarray, shape: Tuple[int, int]
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

    image_nnb = image[y_n, x_n]
    return image_nnb.reshape(shape).astype(image.dtype)


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
    weights = np.array(
        [(1 - dx) * (1 - dy), dx * (1 - dy), (1 - dx) * dy, dx * dy]
    )

    # Calculate the pixel intensity
    neighbors = np.array([p1, p2, p3, p4])
    pixel = weights @ neighbors.T
    return pixel


def gmt_bilinear_interpolation(
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

    # Ensure that the coordinates are inside the image
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
        """R Cubic B-spline function

        Params:
            s: Input array
        Returns:
            R: R function values
        """

        R = (
            np.power(np.maximum(s + 2, 0), 3)
            - 4 * np.power(np.maximum(s + 1, 0), 3)
            + 6 * np.power(np.maximum(s, 0), 3)
            - 4 * np.power(np.maximum(s - 1, 0), 3)
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
        [R(m - dx) * R(dy - n) for m in range(-1, 3) for n in range(-1, 3)]
    )

    # Pad the image
    padded_image = np.pad(image, ((2, 2), (2, 2)), mode='edge')

    # Compute the interpolated pixels
    pixels = np.array(
        [
            padded_image[y0 + n + 2, x0 + m + 2]
            for n in range(-1, 3)
            for m in range(-1, 3)
        ]
    )

    # Compute the output image
    out = np.sum(weights * pixels, axis=0).reshape(shape).astype(image.dtype)
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
            -dx * (dx - 1) * (dx - 2) / 6,
            (dx + 1) * (dx - 1) * (dx - 2) / 2,
            -dx * (dx + 1) * (dx - 2) / 2,
            dx * (dx + 1) * (dx - 1) / 6,
        ]
    )

    # Compute interpolation weights
    w = np.array(
        [
            -dy * (dy - 1) * (dy - 2) / 6,
            (dy + 1) * (dy - 1) * (dy - 2) / 2,
            -dy * (dy + 1) * (dy - 2) / 2,
            dy * (dy + 1) * (dy - 1) / 6,
        ]
    )

    # Pad the image
    padded_image = np.pad(image, ((2, 2), (2, 2)), mode='constant')

    # Compute the interpolated pixels
    pixels = np.array(
        [
            padded_image[y0 + n, x0 + 2 + i]
            for n in range(1, 5)
            for i in range(-1, 3)
        ]
    )

    # Compute the Lagrange interpolation
    L = np.array(
        [np.sum(L_w * pixels[n * 4 : (n + 1) * 4], axis=0) for n in range(4)]
    )

    # Compute the output image
    out = np.sum(w * L, axis=0).reshape(shape).astype(image.dtype)
    return out


interpolations_fucs = {
    'nearest': gmt_nearest_neighbor_interpolation,
    'bilinear': gmt_bilinear_interpolation,
    'bicubic': gmt_bicubic_interpolation,
    'lagrange': gmt_lagrange_interpolation,
}


def open_image(path: Path):
    """Abre uma imagem monocromática e retorna uma matriz numpy.
    Parâmetros:
        path: caminho para a imagem
    Retorno:
        Imagem como uma matriz numpy.
    """
    image = io.imread(path)
    assert len(image.shape) == 2, f'A Imagem {path.name} não é monocromática.'
    return image


def save(image: np.ndarray, output_path: Path):
    """Salva uma imagem no formato PNG.
    Parâmetros:
        image: imagem
        output_path: caminho para a imagem
    """
    if image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)

    io.imsave(output_path, image, check_contrast=False)


def gmt_rotate(image: np.ndarray, angle: float, i_method: str) -> np.ndarray:
    """Rotate an image by an angle.

    Parameters:
        image: input image
        angle: rotation angle (in degrees)
        i_method: interpolation method
    Returns:
        t_image: rotated image
    """

    # Convert the angle to radians
    angle = np.deg2rad(angle)

    R = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ])

    h, w = image.shape

    # Calculate the coordinates of the output image
    corners = np.array([
        [0, 0, 1],
        [0, h - 1, 1],
        [w - 1, h - 1, 1],
        [w - 1, 0, 1]
    ])

    corners_rotated = R.T @ corners.T
    corners_rotated = corners_rotated[:2, :]
    x_min, y_min = corners_rotated.min(axis=1)
    x_max, y_max = corners_rotated.max(axis=1)

    # Calculate the output image size
    output_shape = np.around((y_max - y_min + 1, x_max - x_min + 1)).astype(np.int32)
    out_rows, out_cols = output_shape

    # Pad the image
    pad_x = (out_cols - w) // 2
    pad_y = (out_rows - h) // 2
    padded_image = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant')

    # Translate the image to the origin
    cx, cy = out_cols / 2, out_rows / 2

    T1 = np.array([
        [1, 0, -cx],
        [0, 1, -cy],
        [0, 0, 1]
    ])

    T2 = np.array([
        [1, 0, cx],
        [0, 1, cy],
        [0, 0, 1]
    ])

    M = T2 @ R @ T1
    M_inv = np.linalg.inv(M)

    interpolate = interpolations_fucs[i_method]
    _, t_coords = gmt_coords(out_rows, out_cols, M_inv)
    image_rotated = interpolate(padded_image, t_coords, output_shape)
    return image_rotated


def gmt_scale(image: np.ndarray, scale: float, i_method: str):
    """Scale an image by a factor.

    Parameters:
        image: input image
        scale: scale factor
        i_method: interpolation method
    Returns:
        t_image: scaled image
    """

    # Calculate the scale matrix
    S = np.array([
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, 1],
    ])

    col, row = image.shape
    new_row, new_col = int(row * scale), int(col * scale)
    S_inv = np.linalg.inv(S)

    interpolate = interpolations_fucs[i_method]

    _, t_coords = gmt_coords(new_row, new_col, S_inv)
    image_scaled = interpolate(image, t_coords, (new_row, new_col))
    return image_scaled


def gmt_resize(
    image: np.ndarray, output_shape: tuple[int, int], i_method: str
) -> np.ndarray:
    """Resize an image to a given shape.

    Parameters:
        image: input image
        output_shape: output shape
        i_method: interpolation method
    Returns:
        t_image: resized image
    """

    # Get the scale factors
    rows, cols = image.shape
    cols_out, rows_out = output_shape
    sx, sy = cols_out / cols, rows_out / rows

    # Get the transformation matrix
    S = np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1],
    ])

    S_inv = np.linalg.inv(S)

    interpolate = interpolations_fucs[i_method]

    _, t_coords = gmt_coords(rows_out, cols_out, S_inv)
    image_resized = interpolate(image, t_coords, (rows_out, cols_out))

    return image_resized


def gmt_cli(
    input_file: Path = Option(
        ...,
        '--input',
        '-i',
        help='Imagem de entrada no formato PNG',
        show_default=False,
    ),
    output_file: Path = Option(
        'output.png',
        '--output',
        '-o',
        help='Imagem de saída no formato PNG',
    ),
    angle: float = Option(
        None,
        '--angle',
        '-a',
        help='Angulo de rotação em graus no sentido anti-horário',
        show_default=False,
    ),
    scale: float = Option(
        None, '--scale', '-e', help='Fator de escala', show_default=False
    ),
    output_shape: tuple[int, int] = Option(
        (None, None),
        '--dim',
        '-d',
        help='Dimensões da imagem de saída em pixels (largura altura)',
        show_default=False,
    ),
    method: str = Option(
        'nearest',
        '--method',
        '-m',
        help='Método de interpolação (nearest, bilinear, bicubic, lagrange)',
    ),
):
    """
    Aplica transformações geométricas em imagens PNG.
    """

    assert method in interpolations_fucs, f'Método {method} não suportado.'
    print(f'Usando o método [b blue]{method}[/] para interpolação.')

    if angle != None:
        print(
            f'Rotacionando a imagem [b green]{input_file.name}[/] em {angle}°.'
        )
        tranformet_image = gmt_rotate(open_image(input_file), angle, method)
    elif scale != None:
        print(
            f'Aplicando escala {scale} na imagem [b green]{input_file.name}[/].'
        )
        tranformet_image = gmt_scale(open_image(input_file), scale, method)
    elif output_shape[0] and output_shape[1]:
        print(
            f'Redimensionando a imagem [b green]{input_file.name}[/] para {output_shape}.'
        )
        tranformet_image = gmt_resize(
            open_image(input_file), output_shape, method
        )
    else:
        print('Nenhuma transformação foi aplicada.')
        return

    print(f'Salvando a imagem transformada em [b green]{output_file.name}[/].')
    save(tranformet_image, output_file)


if __name__ == '__main__':
    run(gmt_cli)
