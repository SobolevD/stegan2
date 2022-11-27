import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import convolve2d
from consts import ALPHA, BETA, KEY, M, SIGMA
from utils.embedding import additional_embedding
from utils.fourier import get_fft_image, get_phase_matrix, get_abs_matrix, get_complex_matrix, get_inverse_fft_image
from utils.in_out import read_image, write_image
from utils.snipping import get_H_zone, merge_pictures_H_zone
from watermark import generate_watermark, get_rho, builtin_watermark


def get_betta(c: np.ndarray) -> np.ndarray:
    window = np.ones(shape=(9, 9)) / 81
    mo = convolve2d(c, window, mode='same', boundary='fill')
    mo_2 = np.ndarray(shape=c.shape)
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            tmp = convolve2d((c[i - 4 if i > 3 else 0:
                                      i + 5 if i < (c.shape[0] - 3) else c.shape[0],
                                    j - 4 if j > 3 else 0:
                                    j + 5 if j < (c.shape[1] - 3) else c.shape[1]]
                                    - mo[i, j]) ** 2, window, mode='same', boundary='fill')
            mo_2[i, j] = tmp[i if i < 3 else 4, j if j < 3 else 4]
    res = np.sqrt(mo_2) * 2
    return res / res.max()


def get_optimal_alpha(f, abs_fft_container, phase_fft_container, watermark):
    params = {}

    rho = 0.0
    alpha = 0.6
    while rho < 0.9 or alpha < 1.0:
        H_zone_watermark = additional_embedding(f, BETA, watermark, alpha)

        merged_abs_picture = merge_pictures_H_zone(abs_fft_container, H_zone_watermark)
        complex_matrix = get_complex_matrix(merged_abs_picture, phase_fft_container)
        processed_image = get_inverse_fft_image(complex_matrix)
        write_image(processed_image, 'resource/bridge_processed_tmp.png')

        processed_image = read_image('resource/bridge_processed_tmp.png')
        fft_p_image = get_fft_image(processed_image)
        abs_fft_p_image = get_abs_matrix(fft_p_image)

        H_zone_p = get_H_zone(abs_fft_p_image)
        changed_watermark = builtin_watermark(H_zone_p, f, alpha)
        rho = get_rho(watermark, changed_watermark)

        psnr = cv2.PSNR(watermark, changed_watermark)

        if rho > 0.9:
            params[psnr] = alpha

        print(f'𝜌: {rho}, α: {alpha}, PSNR: {psnr}')
        alpha += 0.02

    min_psnr = min(params.keys())
    max_alpha = params[min_psnr]
    print(f'Result: α: {max_alpha}, Min PSNR: {min_psnr}')
    return max_alpha


def output(C, Cw):
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(C, cmap='gray')
    ax.set_title('Initial')
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(np.real(Cw), cmap='gray')
    ax.set_title('C_w')
    plt.show()

if __name__ == '__main__':

    container = read_image('resource/bridge.tif')

    # 1. Реализовать генерацию ЦВЗ 𝛺 как псевдослучайной последовательности заданной длины из чисел,
    # распределённых по нормальному закону
    H_zone_length = int(container.shape[0] * 0.5) * int(container.shape[1] * 0.5)
    watermark, _ = generate_watermark(H_zone_length, M, SIGMA, KEY)

    # 2. Реализовать трансформацию исходного контейнера к пространству признаков
    fft_container = get_fft_image(container)
    abs_fft_container = get_abs_matrix(fft_container)
    phase_fft_container = get_phase_matrix(fft_container)

    # 3. Осуществить встраивание информации аддитивным методом встраивания.
    # Значения параметравстраивания устанавливается произвольным образом.
    H_zone = get_H_zone(abs_fft_container)
    watermark = watermark.reshape(H_zone.shape)
    H_zone_watermark = additional_embedding(H_zone, BETA, watermark, ALPHA)

    # 4. Сформировать носитель информации при помощи обратного преобразования
    # от матрицы признаков к цифровому сигналу.  Сохранить его на диск.
    merged_abs_picture = merge_pictures_H_zone(abs_fft_container, H_zone_watermark)
    complex_matrix = get_complex_matrix(merged_abs_picture, phase_fft_container)
    processed_image = get_inverse_fft_image(complex_matrix)
    write_image(processed_image, 'resource/bridge_processed.tif')

    # 5. Считать носитель информации из файла. Реализовать трансформацию исходного контейнера к пространству признаков
    processed_image2 = read_image('resource/bridge_processed.tif')
    fft_p_image = get_fft_image(processed_image)
    abs_fft_p_image = get_abs_matrix(fft_p_image)

    # 6. Сформировать оценку встроенного ЦВЗ 𝛺̃неслепым методом (то есть, с использованием
    # матрицы признаков исходного контейнера); выполнить детектирование при помощи функции
    # близости 𝜌(𝛺,𝛺̃) вида (6.11).
    H_zone_p = get_H_zone(abs_fft_p_image)
    changed_watermark = builtin_watermark(H_zone_p, H_zone, ALPHA)
    rho = get_rho(watermark, changed_watermark)

    print(f'𝜌: {rho}')

    # 7. Осуществить автоматический подбор значения параметра встраивания методом перебора
    # с целью обеспечения заданного значения функции близости 𝜌
    get_optimal_alpha(H_zone, abs_fft_container, phase_fft_container, watermark)

    # 8. «Ложное обнаружение»: генерируем 100 случайных последовательностей той же длины, что и 𝛺,
    # и ищем значение функции близости 𝛺 с каждой из них. Строим график, проверяем,
    # удаётся ли выбрать правильную последовательность.
    N = 100
    rho_array = []
    rho_array.append(rho)
    x = np.arange(0, 101)
    for i in range(0, N):
        new_watermark, _ = generate_watermark(H_zone_length, M, SIGMA)
        new_watermark = new_watermark.reshape(H_zone.shape)
        rho = get_rho(new_watermark, changed_watermark)
        rho_array.append(rho)
        if i % 10 == 0:
            print(f'Ready: {i}%')
    ig, ax = plt.subplots()
    ax.plot(x, np.array(rho_array))
    # plt.ylim([0.915, 0.935])
    plt.show()

    betta = get_betta(get_H_zone(container))
    H_zone_watermark = additional_embedding(H_zone, betta, watermark, ALPHA)
    merged_abs_picture = merge_pictures_H_zone(abs_fft_container, H_zone_watermark)
    complex_matrix = get_complex_matrix(merged_abs_picture, phase_fft_container)
    processed_image = get_inverse_fft_image(complex_matrix)

    fft_p_image = get_fft_image(processed_image)
    abs_fft_p_image = get_abs_matrix(fft_p_image)
    H_zone_p = get_H_zone(abs_fft_p_image)
    changed_watermark = builtin_watermark(H_zone_p, H_zone, ALPHA)
    rho = get_rho(watermark, changed_watermark)

    print(f'𝜌: {rho}')