"""
============================================================================
Лабораторная работа №1. Вариант 2.
Задача нескольких коммивояжёров: минимизация времени обхода (mTSP-makespan)

Реализованы два метода СТРОГО по методичке:
  1. Baseline (Greedy + 2-opt): 4 этапа из раздела 3.3
  2. Simulated Annealing: 3 компонента из раздела 3.1

Версия без графиков — только консольный вывод результатов.

Запуск: python Lab_1_FamiliyaIO.py
============================================================================
"""

import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# ИСХОДНЫЕ ДАННЫЕ — ВАРИАНТ 2
# ============================================================

N_CITIES = 12
N_AGENTS = 2
DEPOTS = [0, 6]

DIST = np.array([
    [0.00, 3.03, 1.34, 5.64, 3.02, 2.43, 5.21, 4.03, 4.22, 7.24, 6.48, 0.75],
    [3.03, 0.00, 2.90, 4.39, 2.46, 2.51, 3.88, 6.48, 5.18, 6.31, 7.45, 3.04],
    [1.34, 2.90, 0.00, 6.45, 3.87, 1.23, 5.98, 5.29, 5.51, 8.20, 7.80, 0.63],
    [5.64, 4.39, 6.45, 0.00, 2.62, 6.66, 0.51, 6.44, 3.55, 1.93, 4.77, 6.18],
    [3.02, 2.46, 3.87, 2.62, 0.00, 4.24, 2.19, 4.80, 2.81, 4.34, 5.01, 3.57],
    [2.43, 2.51, 1.23, 6.66, 4.24, 0.00, 6.16, 6.46, 6.33, 8.51, 8.67, 1.82],
    [5.21, 3.88, 5.98, 0.51, 2.19, 6.16, 0.00, 6.29, 3.51, 2.44, 4.96, 5.73],
    [4.03, 6.48, 5.29, 6.44, 4.80, 6.46, 6.29, 0.00, 3.00, 7.12, 3.99, 4.66],
    [4.22, 5.18, 5.51, 3.55, 2.81, 6.33, 3.51, 3.00, 0.00, 4.12, 2.35, 4.96],
    [7.24, 6.31, 8.20, 1.93, 4.34, 8.51, 2.44, 7.12, 4.12, 0.00, 4.31, 7.86],
    [6.48, 7.45, 7.80, 4.77, 5.01, 8.67, 4.96, 3.99, 2.35, 4.31, 0.00, 7.23],
    [0.75, 3.04, 0.63, 6.18, 3.57, 1.82, 5.73, 4.66, 4.96, 7.86, 7.23, 0.00],
])

NON_DEPOT_CITIES = [c for c in range(N_CITIES) if c not in DEPOTS]
N_RUNS = 10
SEEDS = list(range(10))  # 0..9


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def tour_length(tour):
    """Длина замкнутого маршрута."""
    length = 0.0
    for i in range(len(tour)):
        length += DIST[tour[i], tour[(i + 1) % len(tour)]]
    return float(length)


def makespan(routes):
    """Целевая функция: max длины маршрута среди агентов (энергия E)."""
    return max(tour_length(r) for r in routes)


def total_length(routes):
    """Суммарная длина всех маршрутов."""
    return sum(tour_length(r) for r in routes)


def balance_coeff(routes):
    """Коэффициент баланса = std / mean длин маршрутов."""
    lengths = [tour_length(r) for r in routes]
    mean_l = np.mean(lengths)
    if mean_l == 0:
        return 0.0
    return float(np.std(lengths) / mean_l)


def is_feasible(routes):
    """Все города по одному разу, депо на месте, у каждого ≥1 не-депо город."""
    all_cities = []
    for k, route in enumerate(routes):
        if len(route) < 2 or route[0] != DEPOTS[k]:
            return False
        all_cities.extend(route)
    return sorted(all_cities) == list(range(N_CITIES))


def solution_info(routes):
    return {
        'makespan': makespan(routes),
        'total_length': total_length(routes),
        'balance': balance_coeff(routes),
        'feasible': is_feasible(routes),
        'lengths': [tour_length(r) for r in routes],
        'routes': [list(r) for r in routes],
    }


# ============================================================
# МЕТОД 1: BASELINE (GREEDY + 2-OPT) — СТРОГО ПО МЕТОДИЧКЕ
# ============================================================

def sort_city_pairs():
    """Этап 1: сортировка пар городов по возрастанию расстояния."""
    pairs = []
    for i in range(N_CITIES):
        for j in range(i + 1, N_CITIES):
            pairs.append((DIST[i, j], i, j))
    pairs.sort(key=lambda x: x[0])
    return pairs


def greedy_assignment(sorted_pairs):
    """Этап 2: жадное назначение городов агенту с минимальной длиной маршрута."""
    routes = [[DEPOTS[k]] for k in range(N_AGENTS)]
    visited = set(DEPOTS)

    cities_order = []
    for _, i, j in sorted_pairs:
        for c in (i, j):
            if c not in visited and c not in cities_order:
                cities_order.append(c)

    for city in cities_order:
        lengths = [tour_length(r) for r in routes]
        k = int(np.argmin(lengths))
        routes[k].append(city)

    return routes


def two_opt(route):
    """Этап 3: 2-opt для замкнутого маршрута. Депо не трогаем."""
    n = len(route)
    if n < 4:
        return route

    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                old = DIST[route[i-1], route[i]] + DIST[route[j], route[(j+1) % n]]
                new = DIST[route[i-1], route[j]] + DIST[route[i], route[(j+1) % n]]
                if new < old - 1e-10:
                    route[i:j+1] = list(reversed(route[i:j+1]))
                    improved = True
    return route


def inter_route_balancing(routes):
    """Этап 4: попытки переноса городов между агентами для балансировки."""
    improved = True
    while improved:
        improved = False
        lengths = [tour_length(r) for r in routes]
        k_max = int(np.argmax(lengths))
        k_min = int(np.argmin(lengths))

        if k_max == k_min:
            break
        if len(routes[k_max]) <= 2:
            break

        best_impr = 0
        best_idx = -1
        best_pos = -1

        for idx in range(1, len(routes[k_max])):
            city = routes[k_max][idx]
            if city in DEPOTS:
                continue

            new_max = routes[k_max][:idx] + routes[k_max][idx+1:]

            for pos in range(1, len(routes[k_min]) + 1):
                new_min = routes[k_min][:pos] + [city] + routes[k_min][pos:]
                new_ms = max(tour_length(new_max), tour_length(new_min))
                impr = max(lengths) - new_ms

                if impr > best_impr:
                    best_impr = impr
                    best_idx = idx
                    best_pos = pos

        if best_impr > 1e-10:
            city = routes[k_max].pop(best_idx)
            routes[k_min].insert(best_pos, city)
            improved = True

    return routes


def solve_baseline():
    """Baseline — 4 этапа из методички. Полностью детерминированный."""
    sorted_pairs = sort_city_pairs()
    routes = greedy_assignment(sorted_pairs)
    for k in range(N_AGENTS):
        routes[k] = two_opt(routes[k])
    routes = inter_route_balancing(routes)
    return routes


# ============================================================
# МЕТОД 2: SIMULATED ANNEALING — СТРОГО ПО МЕТОДИЧКЕ
# ============================================================

def random_initial_solution():
    """Случайное начальное решение."""
    cities = list(NON_DEPOT_CITIES)
    np.random.shuffle(cities)
    split = np.random.randint(1, len(cities))
    return [
        [DEPOTS[0]] + cities[:split],
        [DEPOTS[1]] + cities[split:],
    ]


def neighbor(routes):
    """
    Компонент 3: генерация соседей через перестановки.
      0: swap внутри маршрута
      1: swap между маршрутами
      2: relocate (перенос между маршрутами)
    """
    new_routes = [list(r) for r in routes]
    op = np.random.randint(0, 3)

    if op == 0:
        k = np.random.randint(0, N_AGENTS)
        if len(new_routes[k]) >= 3:
            idxs = np.random.choice(range(1, len(new_routes[k])), size=2, replace=False)
            i, j = int(idxs[0]), int(idxs[1])
            new_routes[k][i], new_routes[k][j] = new_routes[k][j], new_routes[k][i]

    elif op == 1:
        if len(new_routes[0]) >= 2 and len(new_routes[1]) >= 2:
            i = np.random.randint(1, len(new_routes[0]))
            j = np.random.randint(1, len(new_routes[1]))
            new_routes[0][i], new_routes[1][j] = new_routes[1][j], new_routes[0][i]

    else:
        k_from = np.random.randint(0, N_AGENTS)
        k_to = 1 - k_from
        if len(new_routes[k_from]) >= 3:
            idx = np.random.randint(1, len(new_routes[k_from]))
            city = new_routes[k_from].pop(idx)
            pos = np.random.randint(1, len(new_routes[k_to]) + 1)
            new_routes[k_to].insert(pos, city)

    return new_routes


def estimate_T0(target_acceptance=0.8, n_samples=1000):
    """Подбор T0: ~80% принятия ухудшающих переходов на старте."""
    current = random_initial_solution()
    current_cost = makespan(current)

    deltas = []
    for _ in range(n_samples):
        nb = neighbor(current)
        if not is_feasible(nb):
            continue
        delta = makespan(nb) - current_cost
        if delta > 0:
            deltas.append(delta)
        if np.random.rand() < 0.5:
            current = nb
            current_cost = makespan(nb)

    if not deltas:
        return 10.0
    mean_delta = np.mean(deltas)
    if mean_delta <= 0:
        return 10.0
    return float(-mean_delta / np.log(target_acceptance))


def simulated_annealing(alpha=0.97, T_min=0.001, max_iter=100000, T0=None):
    """
    SA по методичке:
      Компонент 1: T(t) = T0 * alpha^t
      Компонент 2: критерий Метрополиса
      Компонент 3: соседи через перестановки
    """
    if T0 is None:
        T0 = estimate_T0()

    current = random_initial_solution()
    current_cost = makespan(current)

    best = [list(r) for r in current]
    best_cost = current_cost

    T = T0
    iteration = 0
    start_ms = current_cost

    while T >= T_min and iteration < max_iter:
        iteration += 1
        nb = neighbor(current)

        if not is_feasible(nb):
            T = T0 * (alpha ** iteration)
            continue

        E_new = makespan(nb)
        dE = E_new - current_cost

        # Компонент 2: критерий Метрополиса
        if dE < 0:
            accept = True
        else:
            P = np.exp(-dE / T)
            accept = np.random.random() < P

        if accept:
            current = nb
            current_cost = E_new

            if current_cost < best_cost:
                best = [list(r) for r in current]
                best_cost = current_cost

        # Компонент 1: температурный график
        T = T0 * (alpha ** iteration)

    return best, best_cost, start_ms, T0


# ============================================================
# ЭКСПЕРИМЕНТЫ
# ============================================================

def run_experiments():
    results = {'baseline': [], 'sa': []}
    T0_used = []

    print("=" * 70)
    print("ЭКСПЕРИМЕНТЫ: mTSP-makespan, Вариант 2")
    print(f"Города: {N_CITIES}, Агенты: {N_AGENTS}, Депо: {DEPOTS}")
    print("=" * 70)

    for run, seed in enumerate(SEEDS):
        print(f"\n--- Запуск {run+1}/{N_RUNS} (seed={seed}) ---")

        # Baseline
        np.random.seed(seed)
        t0 = time.time()
        bl_routes = solve_baseline()
        t_bl = time.time() - t0
        info = solution_info(bl_routes)
        info['time'] = t_bl
        results['baseline'].append(info)
        print(f"  Baseline: makespan={info['makespan']:.2f}, "
              f"total={info['total_length']:.2f}, bal={info['balance']:.4f}, "
              f"t={t_bl:.3f}s, ok={info['feasible']}")

        # SA
        np.random.seed(seed)
        t0 = time.time()
        sa_routes, sa_cost, start_ms, T0 = simulated_annealing()
        t_sa = time.time() - t0
        info = solution_info(sa_routes)
        info['time'] = t_sa
        results['sa'].append(info)
        T0_used.append(T0)
        print(f"  SA:       makespan={info['makespan']:.2f}, "
              f"total={info['total_length']:.2f}, bal={info['balance']:.4f}, "
              f"t={t_sa:.3f}s, ok={info['feasible']}, "
              f"T0={T0:.2f}, start_ms={start_ms:.2f}")

    print(f"\n  Средняя T0 (автоподбор): {np.mean(T0_used):.2f}")
    return results


# ============================================================
# СТАТИСТИКА
# ============================================================

def print_statistics(results):
    print("\n" + "=" * 70)
    print("СТАТИСТИКА (10 запусков)")
    print("=" * 70)

    for method in ['baseline', 'sa']:
        data = results[method]
        ms = [d['makespan'] for d in data]
        tl = [d['total_length'] for d in data]
        bal = [d['balance'] for d in data]
        tm = [d['time'] for d in data]
        feas = sum(d['feasible'] for d in data)

        name = 'BASELINE (Greedy + 2-opt)' if method == 'baseline' else 'SIMULATED ANNEALING'
        print(f"\n  {name}")
        print(f"  {'Метрика':<18} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
        print("  " + "-" * 60)
        print(f"  {'Makespan':<18} {min(ms):>10.2f} {max(ms):>10.2f} "
              f"{np.mean(ms):>10.2f} {np.std(ms):>10.2f}")
        print(f"  {'Total length':<18} {min(tl):>10.2f} {max(tl):>10.2f} "
              f"{np.mean(tl):>10.2f} {np.std(tl):>10.2f}")
        print(f"  {'Balance (s/L)':<18} {min(bal):>10.4f} {max(bal):>10.4f} "
              f"{np.mean(bal):>10.4f} {np.std(bal):>10.4f}")
        print(f"  {'Time (s)':<18} {min(tm):>10.3f} {max(tm):>10.3f} "
              f"{np.mean(tm):>10.3f} {np.std(tm):>10.3f}")
        print(f"  {'Feasible':<18} {feas}/{len(data)}")

    bl_mean = np.mean([d['makespan'] for d in results['baseline']])
    sa_mean = np.mean([d['makespan'] for d in results['sa']])
    gap = (sa_mean - bl_mean) / bl_mean * 100
    print(f"\n  Gap SA vs Baseline: {gap:+.2f}% "
          f"(отрицательный = SA лучше)")

    print(f"\n  {'ЛУЧШИЕ РЕШЕНИЯ':^50}")
    for method in ['baseline', 'sa']:
        best = min(results[method], key=lambda d: d['makespan'])
        name = 'Baseline' if method == 'baseline' else 'SA'
        print(f"\n  {name}: makespan = {best['makespan']:.2f}")
        print(f"    Длины: {[round(l, 2) for l in best['lengths']]}")
        for k, r in enumerate(best['routes']):
            print(f"    Агент {k} (депо {DEPOTS[k]}): {r}")


# ============================================================
# MAIN
# ============================================================

def main():
    results = run_experiments()
    print_statistics(results)
    print("\n" + "=" * 70)
    print("ГОТОВО.")
    print("=" * 70)


if __name__ == "__main__":
    main()
