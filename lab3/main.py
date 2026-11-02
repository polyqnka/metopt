import sympy as sp


def main():
    x, y, lam = sp.symbols("x y lam", real=True)

    # Вариант 7: минимум f(x, y) при условии x + y = 4
    f = x**2 + y**2
    g = x + y - 4

    # Проверка регулярности ограничения
    grad_g = [sp.diff(g, var) for var in (x, y)]

    print("Целевая функция:", f)
    print("Ограничение g(x, y) = 0:", g)
    print("Градиент ограничения:", grad_g)
    print(
        "Регулярность: градиент ограничения не равен нулевому вектору:",
        any(component != 0 for component in grad_g)
    )

    # Функция Лагранжа
    L = f + lam * g

    # Система необходимых условий
    equations = [
        sp.diff(L, x),
        sp.diff(L, y),
        sp.diff(L, lam)
    ]

    print("\nФункция Лагранжа:", L)
    print("Система уравнений:")

    for equation in equations:
        print(" ", equation, "= 0")

    # Решение системы
    solutions = sp.solve(
        equations,
        (x, y, lam),
        dict=True
    )

    print("\nВсе найденные действительные решения:")

    admissible_points = []

    for number, solution in enumerate(solutions, start=1):
        if not all(
            variable in solution
            for variable in (x, y, lam)
        ):
            continue

        x_value = sp.simplify(solution[x])
        y_value = sp.simplify(solution[y])
        lam_value = sp.simplify(solution[lam])

        if not all(
            value.is_real is not False
            for value in (x_value, y_value, lam_value)
        ):
            continue

        residual = sp.simplify(
            g.subs({x: x_value, y: y_value})
        )

        f_value = sp.simplify(
            f.subs({x: x_value, y: y_value})
        )

        print(f"\nРешение {number}:")
        print("  x* =", x_value)
        print("  y* =", y_value)
        print("  lambda* =", lam_value)
        print("  невязка ограничения =", residual)
        print("  f(x*, y*) =", f_value)

        if residual == 0:
            admissible_points.append(
                (x_value, y_value, lam_value, f_value)
            )

    print(
        "\nЧисло допустимых стационарных точек:",
        len(admissible_points)
    )

    # Проверка глобальности через исключение переменной
    t = sp.symbols("t", real=True)

    reduced_f = sp.expand(
        f.subs({x: t, y: 4 - t})
    )

    canonical_form = sp.expand(
        2 * (t - 2)**2 + 8
    )

    print("\nПроверка глобальности:")
    print(
        "После подстановки y = 4 - x получаем:",
        reduced_f
    )
    print(
        "Эквивалентная форма:",
        canonical_form
    )
    print(
        "Разность выражений:",
        sp.simplify(reduced_f - canonical_form)
    )

    print(
        "Так как квадрат неотрицателен, f >= 8;"
        " равенство достигается при x = y = 2."
    )

    print(
        "При |x| -> бесконечность на допустимой прямой"
        " f -> бесконечность."
    )
    print("Поэтому глобального максимума не существует.")


if __name__ == "__main__":
    main()
