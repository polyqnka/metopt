import unittest
from itertools import product
from math import inf

from main import allocate_resource, load_data, make_variant19_constraint


def brute_force_variant19(
    income: list[list[int | None]],
    resource: int,
    min_per_stage: int,
    max_first_two_sum: int,
) -> tuple[int, list[tuple[int, ...]]]:
    """Независимая проверка полным перебором для варианта 19."""
    n = len(income)
    best_value: int | float = -inf
    best_plans: list[tuple[int, ...]] = []

    for plan in product(range(min_per_stage, resource + 1), repeat=n):
        if sum(plan) != resource:
            continue
        if plan[0] + plan[1] > max_first_two_sum:
            continue
        if any(plan[k] >= len(income[k]) or income[k][plan[k]] is None for k in range(n)):
            continue

        result = sum(income[k][plan[k]] for k in range(n))  # type: ignore[misc]
        if result > best_value:
            best_value = result
            best_plans = [plan]
        elif result == best_value:
            best_plans.append(plan)

    return int(best_value), best_plans


class TestDynamicProgramming(unittest.TestCase):
    def test_variant19_matches_bruteforce_and_manual(self) -> None:
        """Сверка решения варианта 19 с полным перебором и ручными таблицами."""
        data = load_data("data.json")
        income = data["income"]
        resource = data["resource"]
        min_per_stage = data["min_per_stage"]
        max_first_two_sum = data["max_first_two_sum"]

        constraint_fn = make_variant19_constraint(min_per_stage, max_first_two_sum)
        optimum, all_plans, value, actions = allocate_resource(
            income=income,
            resource=resource,
            min_per_stage=min_per_stage,
            allowed_u_fn=constraint_fn,
        )

        bf_value, bf_plans = brute_force_variant19(
            income, resource, min_per_stage, max_first_two_sum
        )

        # Сверка оптимума и планов с перебором
        self.assertEqual(optimum, 47)
        self.assertEqual(optimum, bf_value)
        self.assertEqual([tuple(p) for p in all_plans], bf_plans)
        self.assertEqual(bf_plans, [(2, 2, 6), (2, 3, 5), (3, 2, 5)])

        # Сверка промежуточных таблиц с ручным расчётом
        self.assertEqual({s: value[2][s] for s in range(5, 9)}, {5: 16, 6: 20, 7: 23, 8: 26})
        self.assertEqual({s: value[1][s] for s in range(6, 10)}, {6: 23, 7: 29, 8: 33, 9: 37})
        self.assertEqual(actions[1][8], [2, 3])
        self.assertEqual(actions[0][10], [2, 3])

    def test_1_zero_resource(self) -> None:
        """Тест 1 (п. 8.3): Нулевой объём ресурса."""
        income = [[0], [0], [0]]
        opt, plans, _, _ = allocate_resource(income, resource=0, min_per_stage=0)
        self.assertEqual(opt, 0)
        self.assertEqual(plans, [[0, 0, 0]])

    def test_2_single_project(self) -> None:
        """Тест 2 (п. 8.3): Один проект (этап)."""
        income = [[0, 4, 9, 15]]
        opt, plans, _, _ = allocate_resource(income, resource=3, min_per_stage=0)
        self.assertEqual(opt, 15)
        self.assertEqual(plans, [[3]])

    def test_3_all_zero_incomes(self) -> None:
        """Тест 3 (п. 8.3): Нулевые доходы всех проектов."""
        income = [[0, 0, 0], [0, 0, 0]]
        opt, plans, _, _ = allocate_resource(income, resource=2, min_per_stage=0)
        self.assertEqual(opt, 0)
        self.assertEqual(plans, [[0, 2], [1, 1], [2, 0]])

    def test_4_multiple_optimal_plans(self) -> None:
        """Тест 4 (п. 8.3): Наличие нескольких оптимальных планов (учебный пример из методички)."""
        income = [
            [0, 3, 5, 6, 7],
            [0, 2, 5, 7, 8],
            [0, 1, 4, 6, 9],
        ]
        opt, plans, _, _ = allocate_resource(income, resource=4, min_per_stage=0)
        self.assertEqual(opt, 10)
        self.assertEqual(plans, [[1, 3, 0], [2, 2, 0]])

    def test_5_all_resource_to_one_project(self) -> None:
        """Тест 5 (п. 8.3): Оптимально выделить весь ресурс одному проекту."""
        income = [
            [0, 1, 2, 3],
            [0, 2, 5, 20],
            [0, 1, 2, 3],
        ]
        opt, plans, _, _ = allocate_resource(income, resource=3, min_per_stage=0)
        self.assertEqual(opt, 20)
        self.assertEqual(plans, [[0, 3, 0]])

    def test_6_invalid_inputs(self) -> None:
        """Тест 6 (п. 8.3): Некорректные входные данные."""
        with self.assertRaises(ValueError):
            allocate_resource([[0, 1]], resource=-2)
        with self.assertRaises(ValueError):
            allocate_resource([], resource=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)