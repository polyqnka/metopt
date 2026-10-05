from fractions import Fraction
import pandas as pd

F = Fraction

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


def print_table(A, b, basis, columns, C, title="Симплекс-таблица"):
    """Выводит текущую симплекс-таблицу и возвращает Z, Zj и Δj."""
    Z = {
        col: sum(C.get(basis[i], F(0)) * A[i][j] for i in range(len(A)))
        for j, col in enumerate(columns)
    }

    Z0 = sum(
        C.get(basis[i], F(0)) * b[i]
        for i in range(len(A))
    )

    Delta = {
        col: C.get(col, F(0)) - Z[col]
        for col in columns
    }

    data = []

    for i in range(len(A)):
        data.append(
            [basis[i], C.get(basis[i], F(0)), b[i]] + A[i]
        )

    data.append(["Z", "", Z0] + [Z[col] for col in columns])
    data.append(["Δ = C - Z", "", ""] + [Delta[col] for col in columns])

    table = pd.DataFrame(
        data,
        columns=["Базис", "Cb", "b"] + columns
    )

    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(table.to_string(index=False))

    return Z0, Z, Delta


def pivot(A, b, basis, columns, pivot_row, pivot_col):
    """Выполняет одно симплекс-преобразование относительно разрешающего элемента."""
    pivot_element = A[pivot_row][pivot_col]
    print(
        f"Разрешающий элемент: {pivot_element} "
        f"(строка {basis[pivot_row]}, столбец {columns[pivot_col]})"
    )

    A[pivot_row] = [value / pivot_element for value in A[pivot_row]]
    b[pivot_row] /= pivot_element

    for i in range(len(A)):
        if i == pivot_row:
            continue
        factor = A[i][pivot_col]
        if factor != 0:
            A[i] = [
                A[i][j] - factor * A[pivot_row][j]
                for j in range(len(A[i]))
            ]
            b[i] -= factor * b[pivot_row]

    basis[pivot_row] = columns[pivot_col]


def simplex_phase(A, b, basis, columns, C, phase_name):
    """Решает одну фазу задачи симплекс-методом для максимизации."""
    iteration = 0

    while True:
        iteration += 1
        _, _, Delta = print_table(
            A, b, basis, columns, C,
            f"{phase_name}. Итерация {iteration}"
        )

        positive = [
            (col, Delta[col])
            for col in columns
            if Delta[col] > 0
        ]

        if not positive:
            print("\nВсе Δj <= 0.")
            print("Оптимум данной фазы найден.")
            return A, b, basis

        entering, entering_delta = max(positive, key=lambda item: item[1])
        entering_col = columns.index(entering)
        print(f"\nВводимая переменная: {entering}, Δ = {entering_delta}")

        ratios = []
        for i in range(len(A)):
            coefficient = A[i][entering_col]
            if coefficient > 0:
                ratio = b[i] / coefficient
                ratios.append((ratio, i))
                print(f"{basis[i]}: {b[i]} / {coefficient} = {ratio}")

        if not ratios:
            raise ValueError(f"Задача не ограничена по переменной {entering}")

        min_ratio, pivot_row = min(ratios, key=lambda item: item[0])
        leaving = basis[pivot_row]
        print(f"\nМинимальное отношение: {min_ratio}")
        print(f"Выводимая переменная: {leaving}")

        pivot(A, b, basis, columns, pivot_row, entering_col)


def main():
    """Решает вариант 19 двухфазным симплекс-методом и проверяет ограничения."""
    columns = ["x1", "x2", "x3", "x4", "s1", "s3", "a2", "a3"]

    A = [
        [F(2), F(1), F(0), F(1), F(1), F(0), F(0), F(0)],
        [F(1), F(1), F(1), F(0), F(0), F(0), F(1), F(0)],
        [F(0), F(0), F(1), F(1), F(0), F(-1), F(0), F(1)]
    ]
    b = [F(9), F(7), F(5)]
    basis = ["s1", "a2", "a3"]

    print("=" * 100)
    print("ВАРИАНТ 19")
    print("=" * 100)
    print("max Z = 4x1 + x2 + x3 + 2x4")
    print("2x1 + x2 + x4 <= 9")
    print("x1 + x2 + x3 = 7")
    print("x3 + x4 >= 5")
    print("x1, x2, x3, x4 >= 0")

    C_phase1 = {
        "x1": F(0), "x2": F(0), "x3": F(0), "x4": F(0),
        "s1": F(0), "s3": F(0), "a2": F(-1), "a3": F(-1)
    }

    A, b, basis = simplex_phase(
        A, b, basis, columns, C_phase1, "Фаза I"
    )

    Z_phase1 = sum(
        C_phase1.get(basis[i], F(0)) * b[i]
        for i in range(len(A))
    )

    print("\n" + "=" * 100)
    print("РЕЗУЛЬТАТ ФАЗЫ I")
    print("=" * 100)
    print(f"W* = {-Z_phase1}")

    if Z_phase1 != 0:
        raise ValueError("Исходная задача недопустима: W* != 0")
    print("Исходная задача допустима.")

    artificial = ["a2", "a3"]
    main_columns = [col for col in columns if col not in artificial]
    indices = [columns.index(col) for col in main_columns]
    A_main = [[row[j] for j in indices] for row in A]
    b_main = b.copy()
    basis_main = basis.copy()

    C_main = {
        "x1": F(4), "x2": F(1), "x3": F(1), "x4": F(2),
        "s1": F(0), "s3": F(0)
    }

    print("\n" + "=" * 100)
    print("ПЕРЕХОД К ФАЗЕ II")
    print("=" * 100)
    print(f"Переменные: {main_columns}")
    print(f"Текущий базис: {basis_main}")

    A_main, b_main, basis_main = simplex_phase(
        A_main, b_main, basis_main, main_columns, C_main, "Фаза II"
    )

    solution = {col: F(0) for col in main_columns}
    for i, variable in enumerate(basis_main):
        solution[variable] = b_main[i]

    Z_opt = sum(
        C_main.get(variable, F(0)) * value
        for variable, value in solution.items()
    )

    print("\n" + "=" * 100)
    print("ОПТИМАЛЬНОЕ РЕШЕНИЕ")
    print("=" * 100)
    print(f"x1 = {solution['x1']}")
    print(f"x2 = {solution['x2']}")
    print(f"x3 = {solution['x3']}")
    print(f"x4 = {solution['x4']}")
    print(f"\nZmax = {Z_opt}")

    x1, x2, x3, x4 = (
        solution["x1"], solution["x2"], solution["x3"], solution["x4"]
    )

    constraint_1 = 2 * x1 + x2 + x4
    constraint_2 = x1 + x2 + x3
    constraint_3 = x3 + x4

    print("\n" + "=" * 100)
    print("ПРОВЕРКА ОГРАНИЧЕНИЙ")
    print("=" * 100)
    print(f"1) 2x1 + x2 + x4 = {constraint_1} <= 9")
    print(f"2) x1 + x2 + x3 = {constraint_2} = 7")
    print(f"3) x3 + x4 = {constraint_3} >= 5")

    for variable in ["x1", "x2", "x3", "x4"]:
        print(f"{variable} = {solution[variable]} >= 0")

    assert constraint_1 <= 9
    assert constraint_2 == 7
    assert constraint_3 >= 5
    for variable in ["x1", "x2", "x3", "x4"]:
        assert solution[variable] >= 0

    print("\nВсе ограничения выполнены.")
    print(f"Итог: x* = ({x1}, {x2}, {x3}, {x4}), Z* = {Z_opt}")


if __name__ == "__main__":
    main()
