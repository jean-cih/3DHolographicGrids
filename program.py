import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties


# Размеры области и сетка
def get_size():
    x = np.arange(-scale, scale, step)
    y = np.arange(-scale, scale, step)

    return np.meshgrid(x, y)


# Расчет интерференционной картины (интенсивности)
def calculate_intensity(X, Y, wavelength, theta_R, theta_S):
    # Переводим углы в радианы
    theta_R_rad = np.deg2rad(theta_R)
    theta_S_rad = np.deg2rad(theta_S)

    # Волновые векторы
    k = 2 * np.pi / wavelength
    k_Rx = k * np.sin(theta_R_rad)
    k_Rz = k * np.cos(theta_R_rad)
    k_Sx = k * np.sin(theta_S_rad)
    k_Sz = k * np.cos(theta_S_rad)

    # Разность фаз
    phase_diff = (k_Rx - k_Sx) * X + (k_Rz - k_Sz) * Y

    # Интенсивность (квадрат суммы амплитуд)
    intensity = (np.cos(phase_diff) + 1) ** 2

    return intensity


def calculate_grating_parameters(wavelength, theta_R, theta_S, n_1, n_2):
    # Переводим углы в радианы
    theta_R_rad = np.deg2rad(theta_R)
    theta_S_rad = np.deg2rad(theta_S)

    # Расчет для воздуха (n_1)
    # Пространственные частоты
    freq_y_air = (np.sin(theta_S_rad) - np.sin(theta_R_rad)) / wavelength
    freq_z_air = (np.cos(theta_S_rad) - np.cos(theta_R_rad)) / wavelength
    freq_air = np.sqrt(freq_z_air ** 2 + freq_y_air ** 2)

    # Пространственные периоды
    d_y_air = 1 / freq_y_air if freq_y_air != 0 else float('inf')
    d_z_air = 1 / freq_z_air if freq_z_air != 0 else float('inf')

    # Угол наклона решетки
    psi_air = (theta_S + theta_R) / 2
    psi_air_rad = np.deg2rad(psi_air)
    d_air = d_y_air * np.cos(psi_air_rad)

    # Угол Брэгга для воздуха
    try:
        theta_bragg_air = np.rad2deg(np.arcsin(wavelength / (2 * n_1 * d_air)))
    except ValueError:
        print("Error: Невозможно вычислить угол Брегга для заданных параметров")
        theta_bragg_air = float('nan')

    # Расчет для среды (n_2)
    # Углы в среде по закону Снеллиуса
    psi_R_rad = np.arcsin(n_1 / n_2 * np.sin(theta_R_rad))
    psi_S_rad = np.arcsin(n_1 / n_2 * np.sin(theta_S_rad))

    # Пространственные частоты в среде
    freq_y_medium = (np.sin(psi_S_rad) - np.sin(psi_R_rad)) * n_2 / wavelength
    freq_z_medium = (np.cos(psi_S_rad) - np.cos(psi_R_rad)) * n_2 / wavelength
    freq_medium = np.sqrt(freq_z_medium ** 2 + freq_y_medium ** 2)

    # Пространственные периоды в среде
    d_y_medium = 1 / freq_y_medium if freq_y_medium != 0 else float('inf')
    d_z_medium = 1 / freq_z_medium if freq_z_medium != 0 else float('inf')

    # Угол наклона решетки в среде
    psi_medium = (np.degrees(psi_S_rad) + np.degrees(psi_R_rad)) / 2
    psi_medium_rad = np.deg2rad(psi_medium)
    d_medium = d_y_medium * np.cos(psi_medium_rad)

    # Угол Брэгга для среды
    try:
        theta_bragg_medium = np.rad2deg(np.arcsin(wavelength / (2 * n_2 * d_medium)))
    except ValueError:
        print("Error: Невозможно вычислить угол Брегга для заданных параметров")
        theta_bragg_medium = float('nan')

    return {
        'air': {
            'period': np.abs(d_air),
            'period_y': np.abs(d_y_air),
            'period_z': np.abs(d_z_air),
            'frequency': np.abs(freq_air * 1000),  # переводим в шт./мм
            'frequency_y': np.abs(freq_y_air * 1000),
            'frequency_z': np.abs(freq_z_air * 1000),
            'tilt_angle': psi_air,
            'bragg_angle': theta_bragg_air
        },
        'medium': {
            'period': np.abs(d_medium),
            'period_y': np.abs(d_y_medium),
            'period_z': np.abs(d_z_medium),
            'frequency': np.abs(freq_medium * 1000),  # переводим в шт./мм
            'frequency_y': np.abs(freq_y_medium * 1000),
            'frequency_z': np.abs(freq_z_medium * 1000),
            'tilt_angle': psi_medium,
            'bragg_angle': theta_bragg_medium
        }
    }


def create_graphs(wavelength, theta_R, theta_S, n_1, n_2):
    # Создание 3D графиков
    fig = plt.figure(figsize=(14, 7))

    # График для воздуха
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot_surface(X, Y, I_air, cmap='viridis', rstride=1, cstride=1, alpha=0.8)
    ax1.set_title(f'Дифракционная решетка в воздухе\nλ = {wavelength/n_1:.2f} (мкм), n = {n_1}, θR = {theta_R}°, θS = {theta_S}°')
    ax1.set_xlabel('Z (мкм)')
    ax1.set_ylabel('Y (мкм)')
    ax1.set_zlabel('Интенсивность')

    ax2 = fig.add_subplot(222)
    contour = ax2.contourf(Y, X, I_air, 20, cmap='viridis')
    ax2.set_title(f'Проекция на плоскость X-Y\nθS = {theta_S}°, θR = {theta_R}°, λ = {wavelength/n_1:.2f} (мкм)')
    ax2.set_xlabel('Y (мкм)')
    ax2.set_ylabel('Z (мкм)')
    plt.colorbar(contour)

    # График для среды
    ax2 = fig.add_subplot(223, projection='3d')
    ax2.plot_surface(X, Y, I_medium, cmap='plasma', rstride=1, cstride=1, alpha=0.8)
    ax2.set_title(f'Дифракционная решетка в среде\nλ = {wavelength/n_2:.2f} (мкм), n = {n_2}, θR = {theta_R}°, θS = {theta_S}°')
    ax2.set_xlabel('Z (мкм)')
    ax2.set_ylabel('Y (мкм)')
    ax2.set_zlabel('Интенсивность')

    ax2 = fig.add_subplot(224)
    contour = ax2.contourf(Y, X, I_medium, 20, cmap='plasma')
    ax2.set_title(f'Проекция на плоскость X-Y\nθS = {theta_S}°, θR = {theta_R}°, λ = {wavelength/n_2:.2f} (мкм)')
    ax2.set_xlabel('Y (мкм)')
    ax2.set_ylabel('Z (мкм)')
    plt.colorbar(contour)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5, top=0.90)

    plt.savefig('results_table.png', bbox_inches='tight', dpi=300)


def input_data():
    # Параметры решетки (можно менять)
    wavelength = float(input("Введите длину волны (мкм): "))
    theta_R = float(input("Введите угол референтной волны (градусы): "))
    theta_S = float(input("Введите угол сигнальной волны (градусы): "))
    n_1 = float(input("Введите показатель преломления первой среды: "))
    n_2 = float(input("Введите показатель преломления второй среды: "))

    return wavelength, theta_R, theta_S, n_1, n_2


def create_table_image(params, n_1, n_2):
    # Создаем фигуру и оси
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')  # Отключаем оси

    # Заголовок таблицы
    title = "Результаты расчетов:"

    # Заголовки столбцов
    col_labels = ["Параметр", f"Воздух (n = {n_1})", f"Среда (n = {n_2})"]

    # Данные таблицы
    table_data = [
        ["Пространственный период d", f"{params['air']['period']:.3f} мкм", f"{params['medium']['period']:.3f} мкм"],
        ["Пространственный период dy", f"{params['air']['period_y']:.3f} мкм", f"{params['medium']['period_y']:.3f} мкм"],
        ["Пространственный период dz", f"{params['air']['period_z']:.3f} мкм", f"{params['medium']['period_z']:.3f} мкм"],
        ["Пространственная частота ν", f"{params['air']['frequency']:.3f} шт/мм", f"{params['medium']['frequency']:.3f} шт/мм"],
        ["Пространственная частота νy", f"{params['air']['frequency_y']:.3f} шт/мм", f"{params['medium']['frequency_y']:.3f} шт/мм"],
        ["Пространственная частота νz", f"{params['air']['frequency_z']:.3f} шт/мм", f"{params['medium']['frequency_z']:.3f} шт/мм"],
        ["Угол падения", f"{params['air']['tilt_angle']:.3f}°", f"{params['medium']['tilt_angle']:.3f}°"],
        ["Угол Брэгга", f"{params['air']['bragg_angle']:.3f}°", f"{params['medium']['bragg_angle']:.3f}°"]
    ]

    # Создаем таблицу
    table = ax.table(cellText=table_data,
                     colLabels=col_labels,
                     loc='center',
                     cellLoc='center',
                     colWidths=[0.3, 0.2, 0.2])

    # Настройка стиля таблицы
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.5, 1.5)  # Масштабирование таблицы

    # Жирный шрифт для заголовков
    for (i, j), cell in table.get_celld().items():
        if i == 0 or j == -1:
            cell.set_text_props(fontproperties=FontProperties(weight='bold'))

    # Заголовок
    plt.title(title, fontsize=14, pad=20)

    # Сохраняем таблицу как изображение
    plt.savefig('results_table.png', bbox_inches='tight', dpi=300)



if __name__ == '__main__':

    scale = 1      # масштаб
    step = 0.01        # шаг

    wavelength, theta_R, theta_S, n_1, n_2 = input_data()

    # Создание сетки
    X, Y = get_size()

    # Расчет интенсивности для воздуха
    I_air = calculate_intensity(X, Y, wavelength, theta_R, theta_S)

    # Расчет интенсивности для среды (учитываем показатель преломления)
    I_medium = calculate_intensity(X, Y, wavelength / n_2, theta_R, theta_S)

    # Расчет параметров решетки
    params = calculate_grating_parameters(wavelength, theta_R, theta_S, n_1, n_2)

    create_table_image(params, n_1, n_2)

    create_graphs(wavelength, theta_R, theta_S, n_1, n_2)

    plt.show()
