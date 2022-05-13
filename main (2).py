import scipy.stats as stat
import matplotlib.pyplot as plt
import numpy as np
import math
import itertools

np.set_printoptions(precision=4)

def print_statistics(name_distr, data):
    size = np.size(data)
    print(f'Рапсределение --------- {name_distr}\n')
    if name_distr != "Коши":
        print(f'Выборочное среднее: {np.mean(data)}')
        print(f'Дисперсия: {np.var(data)}')
        print(f'Медиана: {(data[size // 2 - 1] + data[size // 2]) / 2}')
        print(f'1/4 и 3/4 квантили: {np.quantile(data, 1/4)}, {np.quantile(data, 3/4)}')
        print(f'Исправленная дисперсия: {np.var(data) * n / (n - 1)}')
        print(f'Коэф-т ассиметрии: {stat.kstat(data, 3) / stat.kstat(data, 2) ** 1.5}')
        print(f'Коэф-т эксцесса: {stat.moment(data, 4) / stat.moment(data, 2) ** 2}')
        print(f'Коэф-т вариации: { np.var(data)**0.5 / np.mean(data)}\n\n')
    else:
        print(f'Медиана: {(data[size // 2 - 1] + data[size // 2]) / 2}')
        print(f'1/4 и 3/4 квантили: {np.quantile(data, 1 / 4)}, {np.quantile(data, 3 / 4)}\n\n')

def conf_interval_expectation(data, alpha, var):
    point_expectation = np.mean(data)
    u_half_alpha = -stat.norm.ppf(alpha / 2)

    mu_lower = point_expectation - u_half_alpha * (var / n) ** 0.5
    mu_higher = point_expectation + u_half_alpha * (var / n) ** 0.5
    print(f"Выборочное среднее: {np.mean(data)}")
    print(f'Нижняя граница среднего: {mu_lower}')
    print(f'Верхняя граница среднего: {mu_higher}\n')
    return mu_lower, mu_higher

def conf_interval_variance(data, n, alpha):
    s2 = np.var(data) * n / (n - 1)
    df = n - 1
    chi2_half_alpha = stat.chi2.ppf(alpha / 2, df)
    chi2_one_minus_half_alpha = stat.chi2.ppf(1 - alpha / 2, df)

    sigma2_lower = (n - 1) * s2 / chi2_one_minus_half_alpha
    sigma2_higher = (n - 1) * s2 / chi2_half_alpha
    print(f"Выборочная дисперсия: {np.var(data)}")
    print(f'Нижняя граница дисперсии: {sigma2_lower}')
    print(f'Верхняя граница дисперсии: {sigma2_higher}\n')
    return sigma2_lower, sigma2_higher


def conf_interval_expectation_student(data, n, alpha):
    point_expectation = np.mean(data)
    s2 = np.std(data) * n / (n - 1)
    t_half_alpha = -stat.t.ppf(q=alpha / 2, df=n - 1)

    mu_lower = point_expectation - t_half_alpha * (s2 / n) ** 0.5
    mu_higher = point_expectation + t_half_alpha * (s2 / n) ** 0.5
    #print("Gauss expectation with no known variance")
    print(f'Нижняя граница среднего: {mu_lower}')
    print(f'Верхняя граница среднего: {mu_higher}\n')
    return mu_lower, mu_higher


def in_range(x, interval):
    a, b = interval
    if a <= x <= b:
        return True
    return False


def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def group_data(data, k):
    x0 = math.floor(min(data))
    xn = math.ceil(max(data))
    bounds = [x0 + i * (xn - x0) / k for i in range(k + 1)]
    intervals = list(pairwise(bounds))
    invervals_mids = [(a + b) / 2 for (a, b) in intervals]
    grouped = np.zeros(np.size(data))
    for i in range(np.size(data)):
        entry_group = [j for j in range(k) if in_range(data[i], intervals[j])][0]
        grouped[i] = invervals_mids[entry_group]

    _, counts = np.unique(grouped, return_counts=True)
    scaled_counts = [x / sum(counts) for x in counts]
    plt.bar(invervals_mids, [x for x in scaled_counts],
            width=bounds[1] - bounds[0],
            edgecolor='black')
    plt.title("Grouped gauss")
    plt.show()
    return grouped

##############################
n = 200
a = 5
b = 7

gauss_exp = 17
gauss_var = 22
gauss = stat.norm.rvs(loc=gauss_exp, scale=gauss_var ** 0.5, size=n)
plt.hist(gauss)
plt.title(f'Гаусс $n = {n}$, $\mu = {gauss_exp}$, $\sigma^2 = {gauss_var}$')
plt.show()
print_statistics("Гаусс", gauss)

poisson_param = 9
poisson = np.random.poisson(poisson_param, n)
plt.hist(poisson)
plt.title(f'Пуассон $n = {n}$, $\lambda = {poisson_param}$')
plt.show()
print_statistics("Пуассон", poisson)

exp_param = 3
exp = np.random.exponential(1 / exp_param, n)
plt.hist(exp)
plt.title(f'Экспоненциальное $n = {n}$, $\lambda = {exp_param}$')
plt.show()
print_statistics("Экспоненциальное", exp)

cauchy_shift = 0
cauchy_scale = 2
cauchy = stat.cauchy.rvs(loc=cauchy_shift, scale = cauchy_scale, size=n)
plt.hist(cauchy)
plt.title(f'Коши $n = {n}$, $x_0 = {cauchy_shift}$, $\gamma = {cauchy_scale}$')
plt.show()
print_statistics("Коши", cauchy)

uniform = np.random.uniform(low=a, high=b, size=n)
plt.hist(uniform)
plt.title(f'Равномерное $n = {n}$, $a = {a}$, $b = {b}$')
plt.show()
print_statistics("Равномерное", uniform)

##############################################################
alpha = 0.05
gauss_exp = 4
gauss_var = 9

# 2nd param is standard deviation, not variance
gauss = stat.norm.rvs(loc=gauss_exp, scale=gauss_var ** 0.5, size=n)

print(f"error: {alpha}")
mu_lower, mu_higher = conf_interval_expectation(gauss, alpha, gauss_var)
print("Доверительный интервал для ожидания с фактической сигмой")
sigma2_lower, sigma2_higher = conf_interval_variance(gauss, n, alpha)

plt.hist(gauss, edgecolor = 'black')
plt.title("Ungrouped gauss")
plt.show()

##############################################################
k = 8
print(f'\n\n-----------Grouped gauss, k = {k}------------')
grouped = group_data(gauss, k)

mu_lower_group, mu_higher_group = conf_interval_expectation(grouped, alpha, gauss_var)
sigma2_lower_group, sigma2_higher_group = conf_interval_variance(grouped, n, alpha)
mu_lower_student, mu_higher_student = conf_interval_expectation_student(gauss, k, alpha)

print("\n-----------Grouped gauss expectation w/ actual variance----------")
print(f"point estimation: {np.mean(grouped)}")
print(f'Нижняя граница ожидания: {mu_lower_group}')
print(f'Верхняя граница ожидания: {mu_higher_group}\n')

print("\n-----------Ungrouped gauss expectation w/ actual variance-----------")
print(f"point estimation: {np.mean(gauss)}")
print(f'Нижняя граница ожидания: {mu_lower}')
print(f'Верхняя граница ожидания: {mu_higher}\n')
