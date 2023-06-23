from pathlib import Path

import numpy as np
from rich import print
from skimage import io, transform
from typer import Option, run

from gmt import (
    gmt_coords,
    gmt_warp,
    gmt_nearest_neighbor_interpolation,
    gmt_bilinear_interpolation,
    gmt_bicubic_interpolation,
    gmt_lagrange_interpolation,
)


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


def gmt_rotate(image: np.ndarray, angle: float, i_method: str):
    """
    Rotaciona uma imagem no sentido anti-horário.
    """
    # image_rotated = transform.rotate(
    #     image,
    #     angle,
    #     resize=True,
    #     center=None,
    #     order=3,
    #     mode='constant',
    #     cval=1,
    # )

    # Convert the angle to radians
    angle = np.deg2rad(angle)

    R = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )

    # Translate the image to the origin
    h, w = image.shape
    cx, cy = w // 2, h // 2
    T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])

    M = np.linalg.inv(T1) @ R @ T1

    M_inv = np.linalg.inv(M)

    interpolate = interpolations_fucs[i_method]

    _, t_coords = gmt_coords(*image.shape, M_inv)

    image_rotated = interpolate(image, t_coords, image.shape)
    return image_rotated


def gmt_scale(image: np.ndarray, scale: float, i_method: str):
    """
    Aplica uma escala em uma imagem.
    """
    # image_scaled = transform.rescale(
    #     image, scale, order=3, mode='constant', cval=1, anti_aliasing=False
    # )

    S = np.array(
        [
            [scale, 0, 0],
            [0, scale, 0],
            [0, 0, 1],
        ]
    )
    col, row = image.shape
    new_row, new_col = int(row * scale), int(col * scale)
    S_inv = np.linalg.inv(S)

    interpolate = interpolations_fucs[i_method]

    _, t_coords = gmt_coords(new_row, new_col, S_inv)
    image_scaled = interpolate(image, t_coords, (new_row, new_col))
    return image_scaled


def gmt_resize(
    image: np.ndarray, output_shape: tuple[int, int], i_method: str
):
    """
    Redimensiona uma imagem.
    """
    # image_resized = transform.resize(
    #     image,
    #     output_shape,
    #     order=3,
    #     mode='constant',
    #     cval=1,
    #     clip=True,
    #     anti_aliasing=False,
    # )

    # Get the scale factors
    rows, cols = image.shape
    cols_out, rows_out = output_shape
    sx, sy = cols_out / cols, rows_out / rows

    # Get the transformation matrix
    S = np.array(
        [
            [sx, 0, 0],
            [0, sy, 0],
            [0, 0, 1],
        ]
    )

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
