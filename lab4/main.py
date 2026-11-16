import json
from math import inf
from pathlib import Path
from typing import Callable, Optional


def load_data(filepath: str = "data.json") -> dict:
    """Загружает исходные данные задачи из JSON-файла."""
    path = Path(filepath)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def allocate_resource(
    income: list[list[Optional[int | float]]],
    resource: int,
    min_per_stage: int = 0,
    allowed_u_fn: Optional[Callable[[int, int, int], list[int]]] = None,
) -> tuple[int | float, list[list[int]], list[list[int | float]], list[list[list[int]]]]:
    """
    Решает задачу полного распределения дискретного ресурса методом ДП.

    Параметры:
        income: таблица эффектов income[k][u] для этапа k и управления u
                (None означает, что данное значение u недопустимо по таблице).
        resource: общий объём распределяемого ресурса (B).
        min_per_stage: минимальный объём ресурса на каждый этап.
        allowed_u_fn: необязательная функция allowed_u_fn(k, s, resource),
                      возвращающая список допустимых управлений u на этапе k
                      в состоянии s (с учётом дополнительных ограничений).

    Возвращает:
        optimum: оптимальное значение целевой функции V_1(resource);
        all_plans: список всех оптимальных планов;
        value: таблица функций Беллмана value[k][s];
        actions: таблица множеств оптимальных управлений actions[k][s].
    """
    if not isinstance(resource, int) or resource < 0:
        raise ValueError("Объём ресурса должен быть неотрицательным целым числом.")
    if not income:
        raise ValueError("Необходимо задать хотя бы один этап (объект).")

    n = len(income)
    value: list[list[int | float]] = [[-inf] * (resource + 1) for _ in range(n + 1)]
    actions: list[list[list[int]]] = [[[] for _ in range(resource + 1)] for _ in range(n)]

    # Терминальное условие: после последнего этапа остаток должен быть равен 0.
    value[n][0] = 0

    # Обратный проход: от последнего этапа (n - 1) к первому (0).
    for k in range(n - 1, -1, -1):
        max_table_u = len(income[k]) - 1
        for s in range(resource + 1):
            if allowed_u_fn is not None:
                candidates_u = allowed_u_fn(k, s, resource)
            else:
                # Стандартное ограничение: каждому этапу минимум min_per_stage
                max_u = min(s - (n - 1 - k) * min_per_stage, max_table_u)
                candidates_u = list(range(min_per_stage, max_u + 1))

            for u in candidates_u:
                if u < 0 or u > s or u > max_table_u:
                    continue
                gain = income[k][u]
                if gain is None:
                    continue
                future = value[k + 1][s - u]
                if future == -inf:
                    continue

                candidate = gain + future
                if candidate > value[k][s]:
                    value[k][s] = candidate
                    actions[k][s] = [u]
                elif candidate == value[k][s]:
                    actions[k][s].append(u)

    optimum = value[0][resource]
    if optimum == -inf:
        raise ValueError("Задача не имеет допустимых решений при заданных ограничениях.")

    # Прямой проход: восстановление всех оптимальных планов.
    all_plans: list[list[int]] = []

    def backtrack(k: int, s: int, current_plan: list[int]) -> None:
        if k == n:
            if s == 0:
                all_plans.append(current_plan.copy())
            return
        for u in actions[k][s]:
            current_plan.append(u)
            backtrack(k + 1, s - u, current_plan)
            current_plan.pop()

    backtrack(0, resource, [])
    return optimum, all_plans, value, actions


def make_variant19_constraint(min_per_stage: int, max_first_two_sum: int) -> Callable[[int, int, int], list[int]]:
    """
    Формирует функцию допустимых управлений U_k(s) для варианта 19:
    x_k >= 1 для всех k, x_1 + x_2 <= 5.
    """
    def allowed_u(k: int, s: int, total_resource: int) -> list[int]:
        if k == 0:
            # Этап 1 (s = 10): x_1 >= 1, x_1 + x_2 <= 5 => x_1 <= 4
            if s != total_resource:
                return []
            max_u = max_first_two_sum - min_per_stage
            return list(range(min_per_stage, max_u + 1))
        if k == 1:
            # Этап 2 (s = 10 - x_1): уже потрачено x_1 = total_resource - s.
            # Требуется x_1 + u_2 <= 5 => u_2 <= 5 - (10 - s) = s - 5.
            allocated_first = total_resource - s
            if allocated_first < min_per_stage or allocated_first > max_first_two_sum - min_per_stage:
                return []
            max_u = max_first_two_sum - allocated_first
            return list(range(min_per_stage, max_u + 1))
        if k == 2:
            # Этап 3: забирает весь остаток s, если он допустим (5 <= s <= 8)
            min_s3 = total_resource - max_first_two_sum
            max_s3 = total_resource - 2 * min_per_stage
            if min_s3 <= s <= max_s3:
                return [s]
            return []
        return []

    return allowed_u


def verify_plans(
    plans: list[list[int]],
    expected_optimum: int | float,
    income: list[list[Optional[int | float]]],
    resource: int,
    min_per_stage: int,
    max_first_two_sum: int,
) -> None:
    """Проверяет выполнение всех ограничений и совпадение целевой функции."""
    assert len(plans) > 0, "Список оптимальных планов пуст."
    for plan in plans:
        assert len(plan) == len(income), f"Неверная длина плана: {plan}"
        assert sum(plan) == resource, f"Сумма бригад не равна {resource}: {plan}"
        assert all(x >= min_per_stage for x in plan), f"Нарушен минимум бригад на участок: {plan}"
        assert plan[0] + plan[1] <= max_first_two_sum, (
            f"Нарушено ограничение общежития (x1 + x2 <= {max_first_two_sum}): {plan}"
        )
        total_gain = sum(income[k][plan[k]] for k in range(len(plan)))  # type: ignore[operator]
        assert total_gain == expected_optimum, (
            f"Выработка плана {plan} ({total_gain}) не совпадает с оптимумом ({expected_optimum})"
        )


def print_dp_tables(
    stages: list[str],
    value: list[list[int | float]],
    actions: list[list[list[int]]],
) -> None:
    """Выводит таблицы функций Беллмана и оптимальных управлений для сверки с ручным счётом."""
    print("\n=== ТАБЛИЦЫ ДИНАМИЧЕСКОГО ПРОГРАММИРОВАНИЯ (ОБРАТНЫЙ ПРОХОД) ===")
    for k in range(len(stages) - 1, -1, -1):
        print(f"\nЭтап {k + 1} ({stages[k]}):")
        print(f"{'Остаток s':<12} | {'V_' + str(k + 1) + '(s)':<10} | {'A_' + str(k + 1) + '(s)'}")
        print("-" * 42)
        for s in range(len(value[k])):
            if value[k][s] != -inf:
                print(f"{s:<12} | {value[k][s]:<10} | {actions[k][s]}")


def main() -> None:
    data = load_data("data.json")
    income = data["income"]
    resource = data["resource"]
    stages = data["stages"]
    min_per_stage = data["min_per_stage"]
    max_first_two_sum = data["max_first_two_sum"]

    constraint_fn = make_variant19_constraint(min_per_stage, max_first_two_sum)
    optimum, all_plans, value, actions = allocate_resource(
        income=income,
        resource=resource,
        min_per_stage=min_per_stage,
        allowed_u_fn=constraint_fn,
    )

    verify_plans(all_plans, optimum, income, resource, min_per_stage, max_first_two_sum)

    print(f"Вариант {data['variant']}: «{data['title']}»")
    print_dp_tables(stages, value, actions)

    print("\n=== ИТОГОВЫЙ ОТВЕТ ===")
    print(f"1. Значение: {optimum}")
    print(f"2. Планы ({len(all_plans)} шт.):")
    for idx, plan in enumerate(all_plans, start=1):
        gains = [income[k][plan[k]] for k in range(len(plan))]
        print(
            f"   План {idx}: {tuple(plan)} -> "
            f"выработка по участкам: {gains[0]} + {gains[1]} + {gains[2]} = {sum(gains)}"
        )
    print(f"3. Единственный ли план: {'Да' if len(all_plans) == 1 else 'Нет (найдено 3 оптимальных плана)'}")
    print("4. Проверка: все ограничения (x1 + x2 + x3 = 10, x_i >= 1, x1 + x2 <= 5) выполнены.")


if __name__ == "__main__":
    main()