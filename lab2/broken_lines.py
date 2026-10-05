import time

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


def parse_function(function_str):
    """
    Преобразует строковое представление функции в функцию Python.

    Parameters
    ----------
    function_str : str
        Строка с функцией, например:
        "f(x) = 10 + x**2 - 10*cos(2*pi*x)".

    Returns
    -------
    function : callable
        Функция, которую можно вычислять для числовых значений x.
    expression : sympy.Expr
        Символьное представление функции.
    """
    x = sp.symbols("x")

    function_str = function_str.strip()

    if "=" in function_str:
        function_str = function_str.split("=", 1)[1].strip()

    expression = sp.sympify(
        function_str,
        locals={
            "x": x,
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "asin": sp.asin,
            "acos": sp.acos,
            "atan": sp.atan,
            "exp": sp.exp,
            "log": sp.log,
            "sqrt": sp.sqrt,
            "pi": sp.pi,
            "e": sp.E,
            "abs": sp.Abs,
        },
    )

    function = sp.lambdify(x, expression, "numpy")

    return function, expression


def broken_lines(function_str, a, b, eps, L):
    """
    Находит глобальный минимум функции методом ломаных
    Пиявского–Шуберта.

    Parameters
    ----------
    function_str : str
        Строковое представление исследуемой функции.
    a : float
        Левая граница интервала.
    b : float
        Правая граница интервала.
    eps : float
        Требуемая точность.
    L : float
        Константа Липшица.

    Returns
    -------
    x_best : float
        Найденная точка минимума.
    f_best : float
        Значение функции в найденной точке.
    iterations : int
        Количество выполненных итераций.
    evaluations : int
        Количество вычислений функции.
    elapsed_time : float
        Время работы алгоритма в секундах.
    points : list
        Список исследованных точек и значений функции.
    lower_bound : float
        Минимальная нижняя оценка функции на последней итерации.
    expression : sympy.Expr
        Символьное представление функции.
    """
    function, expression = parse_function(function_str)

    start_time = time.perf_counter()

    y1 = float(function(a))
    y2 = float(function(b))

    points = [
        (float(a), y1),
        (float(b), y2),
    ]

    iterations = 0

    while True:
        points.sort(key=lambda point: point[0])

        candidates = []

        for i in range(len(points) - 1):
            x_left, y_left = points[i]
            x_right, y_right = points[i + 1]

            x_new = (
                (x_left + x_right) / 2
                + (y_left - y_right) / (2 * L)
            )

            x_new = max(x_left, min(x_new, x_right))

            lower = y_left - L * (x_new - x_left)

            candidates.append(
                (lower, x_new)
            )

        lower_bound, x_new = min(
            candidates,
            key=lambda item: item[0]
        )

        x_best, f_best = min(
            points,
            key=lambda point: point[1]
        )

        if f_best - lower_bound <= eps:
            break

        y_new = float(function(x_new))
        points.append((x_new, y_new))

        iterations += 1

    elapsed_time = time.perf_counter() - start_time
    evaluations = len(points)

    return (
        x_best,
        f_best,
        iterations,
        evaluations,
        elapsed_time,
        points,
        lower_bound,
        expression,
    )


def plot_result(
    function_str,
    a,
    b,
    L,
    points,
    x_best,
    f_best,
):
    """
    Строит график исходной функции, нижней ломаной,
    исследованных точек и найденного минимума.

    Parameters
    ----------
    function_str : str
        Строковое представление функции.
    a : float
        Левая граница интервала.
    b : float
        Правая граница интервала.
    L : float
        Константа Липшица.
    points : list
        Исследованные точки и значения функции.
    x_best : float
        Найденная точка минимума.
    f_best : float
        Значение функции в найденной точке.
    """
    function, _ = parse_function(function_str)

    x_grid = np.linspace(a, b, 3000)
    y_grid = function(x_grid)

    points = sorted(points, key=lambda point: point[0])

    x_points = np.array([point[0] for point in points])
    y_points = np.array([point[1] for point in points])

    lower_envelope = []

    for x in x_grid:
        values = [
            y_i - L * abs(x - x_i)
            for x_i, y_i in points
        ]

        lower_envelope.append(min(values))

    lower_envelope = np.array(lower_envelope)

    plt.figure(figsize=(12, 7))

    plt.plot(
        x_grid,
        y_grid,
        label="f(x)",
        linewidth=2,
    )

    plt.plot(
        x_grid,
        lower_envelope,
        label="Нижняя ломаная",
        linewidth=1.5,
    )

    plt.scatter(
        x_points,
        y_points,
        s=18,
        label="Точки вычисления",
    )

    plt.scatter(
        [x_best],
        [f_best],
        s=100,
        marker="*",
        label="Найденный минимум",
    )

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Метод ломаных Пиявского–Шуберта")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    function_str = "10 + x**2 - 10*cos(2*pi*x)"

    a = -1.8
    b = 1.7
    eps = 0.01
    L = 70

    print("=" * 60)
    print("МЕТОД ЛОМАНЫХ")
    print("=" * 60)

    print(f"Функция: {function_str}")
    print(f"Интервал: [{a}, {b}]")
    print(f"Точность eps: {eps}")
    print(f"Константа Липшица L: {L}")
    print()

    (
        x_best,
        f_best,
        iterations,
        evaluations,
        elapsed_time,
        points,
        lower_bound,
        expression,
    ) = broken_lines(
        function_str,
        a,
        b,
        eps,
        L,
    )

    print("РЕЗУЛЬТАТ")
    print("-" * 60)

    print(f"x* = {x_best:.8f}")
    print(f"f(x*) = {f_best:.8f}")
    print()
    print(f"Количество итераций: {iterations}")
    print(f"Количество вычислений функции: {evaluations}")
    print(f"Время работы: {elapsed_time:.6f} сек.")
    print()
    print(f"Нижняя оценка: {lower_bound:.8f}")
    print(
        f"Ошибка по нижней оценке: "
        f"{f_best - lower_bound:.8f}"
    )
    print()
    print(
        "Условие точности:",
        f"{f_best - lower_bound:.8f} <= {eps}",
    )

    plot_result(
        function_str,
        a,
        b,
        L,
        points,
        x_best,
        f_best,
    )