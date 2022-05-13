import scipy.stats as stat
import numpy as np
import math
import itertools

alpha = 0.05
k = 8
u_half_alpha = -stat.norm.ppf(alpha / 2)

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
    return grouped


def group_data_bins(data, k):
    x0 = math.floor(min(data))
    xn = math.ceil(max(data))
    bounds = [x0 + i * (xn - x0) / k for i in range(k + 1)]
    intervals = list(pairwise(bounds))
    invervals_mids = [(a + b) / 2 for (a, b) in intervals]
    grouped = np.zeros(np.size(invervals_mids))
    for i in range(np.size(data)):
        entry_group = [j for j in range(k) if in_range(data[i], intervals[j])][0]
        grouped[entry_group] += 1
    return intervals, grouped

def t_test_homogeneous(x, y, n1, n2):
    # n1 = np.size(x)
    # n2 = np.size(y)
    n_x = np.size(x)
    n_y = np.size(y)

    s2 = ((n_x - 1) * np.var(x) + (n_y - 1) * np.var(y)) / (n_x + n_y - 2)
    t = (np.mean(x) - np.mean(y)) / (s2 ** 0.5 * (1 / n_x + 1 / n_y) ** 0.5)
    t_half_alpha = stat.t.ppf(1 - alpha / 2, df=n1 + n2 - 2)
    p_value = 2 * stat.norm.cdf(-np.abs(t))

    print(f'\n\nt statistic value: {t}')
    print(f'critical values: [ -{t_half_alpha},{t_half_alpha}]\n\n')
   # print(f'p-value: {p_value}\n\n')


def t_test_dependent(x, y, n):
    z = y - x
    t = np.mean(z) / (np.std(z) / np.size(n) ** 0.5)
    t_half_alpha = stat.t.ppf(1 - alpha / 2, df=n)
    p_value = 2 * stat.norm.cdf(-np.abs(t))
    print(f'\n\nt statistic value: {t}')
    print(f'critical values: [ -{t_half_alpha},{t_half_alpha}]\n\n')
   # print(f'p-value: {p_value}\n\n')

def empirical_cdf(t, data):
    return sum([1 for x in data if x <= t]) / np.size(data)

def kolmogorov_normality_test(data):
    mean = np.mean(data)
    std = np.std(data)
    dif = [np.abs(empirical_cdf(t, data) - stat.norm.cdf(t, loc=mean, scale=std)) for t in data]
    k_statistic = np.size(data) ** 0.5 * np.max(dif)
    kolmogorov_quant = stat.kstwobign.ppf(1 - alpha)
    p_value = 1 - stat.kstwobign.cdf(k_statistic)
    print(f'\n\nk statistic value: {k_statistic}')
    print(f'critical value: {kolmogorov_quant}\n\n')
    #print(f'p-value: {p_value}\n\n')


def chi2_normality_test(data):
    mean = np.mean(data)
    std = np.std(data)
    intervals, freq = group_data_bins(data, k=10)

    exp = [np.size(data) * (stat.norm.cdf(b, loc=mean, scale=std) - stat.norm.cdf(a, loc=mean, scale=std)) for (a, b) in
           intervals]
    chi2 = np.sum((freq - exp) ** 2 / exp)

    chi2_quant = stat.chi2.ppf(1 - alpha, df=k - 2)  # -2 for 2 estimated parameters (location and scale)
    p_value = 1 - stat.chi2.cdf(chi2, df=k - 2)
    print(f'\n\nchi2 statistic value: {chi2}')
    print(f'critical value: {chi2_quant}\n\n')
    #print(f'p-value: {p_value}\n\n')


def KS_test(x, y):
    cumulated = np.append(x, y)
    n1 = np.size(x)
    n2 = np.size(y)
    dif = [np.abs(empirical_cdf(t, x) - empirical_cdf(t, y)) for t in cumulated]
    k_statistic = np.max(dif)
    crit_value = ((n1 + n2) / (n1 * n2)) ** 0.5 * (-np.log(alpha / 2) * 0.5) ** 0.5
    p_value = 1 - stat.kstwobign.cdf(k_statistic * ((n1 * n2) / (n1 + n2)) ** 0.5)

    print(f'\n\nk statistic value: {k_statistic}')
    print(f'critical value: {crit_value}\n\n')
    #print(f'p-value: {p_value}\n\n')

############Simple hypothesis##############

#####Homogeneity test for normal's expectation
print("--------------t-criterion for expectation-------------------")

n_x_gauss = 200
n_y_gauss = 300

x_gauss = stat.norm.rvs(loc=5, scale=7**0.5,size=n_x_gauss)
y_gauss = stat.norm.rvs(loc=5, scale=9**0.5,size=n_y_gauss)

print("\n___independent samples___\n")
print("\nungrouped\n")
t_test_homogeneous(x_gauss, y_gauss, n_x_gauss, n_y_gauss)

print("\ngrouped\n")
t_test_homogeneous(group_data(x_gauss,k), group_data(y_gauss,k), k, k)

n_x_gauss = 200

x_gauss = stat.norm.rvs(loc=3, scale=12**0.5,size=n_x_gauss)
y = 5*x_gauss + np.random.default_rng().uniform(low= -6, high=6, size=n_x_gauss)

print("\n___dependent samples___\n")
print("\nungrouped\n")
t_test_dependent(x_gauss, y, n_x_gauss)

print("\ngrouped\n")
t_test_dependent(group_data(x_gauss,k), group_data(y,k), k)

########poission

lambda_x = 9
n_x_poisson = 200

x_poisson = stat.poisson.rvs(lambda_x, size=n_x_poisson)

x1_poisson = x_poisson[:n_x_poisson//2]
x2_poisson = x_poisson[n_x_poisson//2:]

t = (np.mean(x1_poisson) - np.mean(x2_poisson)) / (np.mean(x1_poisson) + np.mean(x2_poisson))**0.5

t_left_quant = stat.norm.ppf(alpha/2)
t_right_quant = stat.norm.ppf(1 - alpha/2)
p_value = 2 * stat.norm.cdf(-np.abs(t))

print("\npoisson\n")
print(f'\n\nt statistic value: {t}')
print(f'critical values: [ {t_left_quant},{t_right_quant}]\n\n')
#print(f'p-value: {p_value}\n\n')

##########Homogenety tests for variance########

#####independent gauss data

print("-------------Fisher criterion for variance--------")
print("____independent gauss data_____")
n_x_gauss = 200
n_y_gauss = 300

x_gauss = stat.norm.rvs(loc=5, scale=7**0.5,size=n_x_gauss)
y_gauss = stat.norm.rvs(loc=5, scale=9**0.5,size=n_y_gauss)

fisher_left_quan = stat.f.ppf(alpha / 2, dfn = n_x_gauss-1, dfd = n_y_gauss - 1)
fisher_right_quan = stat.f.ppf(1 - alpha / 2, dfn = n_x_gauss-1, dfd = n_y_gauss - 1)

F = np.var(x_gauss) / np.var(y_gauss)
p_value = stat.f.cdf(F, dfn = n_x_gauss-1, dfd = n_y_gauss - 1)

print(f'\n\nF statistic value: {F}')
print(f'critical values: [ {fisher_left_quan},{fisher_right_quan}]\n\n')
#print(f'p-value: {p_value}\n\n')

######gauss data from one sample

print("______gauss data from one sample______")
n_y_gauss = 300
y_gauss = stat.norm.rvs(loc=5, scale=9**0.5,size=n_y_gauss)
y1_gauss = y_gauss[:n_y_gauss // 2]
y2_gauss = y_gauss[n_y_gauss // 2:]

fisher_left_quan = stat.f.ppf(alpha / 2, dfn = n_y_gauss / 2 - 1, dfd = n_y_gauss / 2 - 1)
fisher_right_quan = stat.f.ppf(1 - alpha / 2, dfn = n_y_gauss / 2 - 1, dfd = n_y_gauss / 2 - 1)

F = np.var(y1_gauss) / np.var(y2_gauss)
p_value = stat.f.cdf(F, dfn = n_y_gauss / 2 - 1, dfd = n_y_gauss / 2 - 1)

print(f'\n\nF statistic value: {F}')
print(f'critical values: [ {fisher_left_quan},{fisher_right_quan}]\n\n')
#print(f'p-value: {p_value}\n\n')

#########Correlation test###########
#######correlated gauss data
print("----------correlation hypothesis---------------")
print("______correlated gauss data_______")
n_x_gauss = 200

x_gauss = stat.norm.rvs(loc=3, scale=12**0.5,size=n_x_gauss)
y = 5 * x_gauss + 0.1*np.random.default_rng().uniform(low= -6, high=6, size=n_x_gauss)
ro, _ = stat.pearsonr(x_gauss, y)

t = ro / (1 - ro**2)**0.5 * (n_x_gauss - 2)**0.5
t_right_quant = stat.t.ppf(1-alpha/2, df=n_x_gauss)
t_left_quant = stat.t.ppf(alpha/2, df=n_x_gauss)
p_value = 2 * stat.norm.cdf(-np.abs(t))

print(f'\n\nt statistic value: {t}')
print(f'critical values: [ {t_left_quant},{t_right_quant}]\n\n')
#print(f'p-value: {p_value}\n\n')

#######uncorrelated gauss data

print("______uncorrelated gauss data______")
n_x_gauss = 200

x_gauss = stat.norm.rvs(loc=3, scale=12**0.5,size=n_x_gauss)
x1_gauss = x_gauss[:n_x_gauss//2]
x2_gauss = x_gauss[n_x_gauss//2:]

ro, _ = stat.pearsonr(x1_gauss, x2_gauss)

t = ro / (1 - ro**2)**0.5 * (n_x_gauss//2 - 2)**0.5
t_right_quant = stat.t.ppf(1-alpha/2, df=n_x_gauss//2)
t_left_quant = stat.t.ppf(alpha/2, df=n_x_gauss//2)
p_value = 2 * stat.norm.cdf(-np.abs(t))

print(f't statistic value: {t}')
print(f'critical values: [ {t_left_quant},{t_right_quant}]\n\n')
#print(f'p-value: {p_value}')

#######Goodness-of-fit tests#####

####chi square test for gauss

print("------------pearson criterion------------")
n_x_gauss = 400
x_gauss = stat.norm.rvs(loc=5, scale=7**0.5,size=n_x_gauss)

print('gauss')
chi2_normality_test(x_gauss)

print('gauss+uniform noise')
x_gauss_uni_noise = x_gauss + np.random.default_rng().uniform(low=-6, high=6, size=n_x_gauss)
chi2_normality_test(x_gauss_uni_noise)

print('gauss+cauchy noise')
# noise must be small as otherwise plug-in estimations for variance will cause zero division
x_gauss_cauchy_noise = x_gauss + 0.01 * stat.cauchy.rvs(loc=0, scale=1, size=n_x_gauss)
chi2_normality_test(x_gauss_cauchy_noise)

######Kolmogorov test

print("-------kolmogorov test---------")
print('gauss')
kolmogorov_normality_test(x_gauss)
print('gauss+uniform noise')
kolmogorov_normality_test(x_gauss_uni_noise)
print('gauss+cauchy noise')
kolmogorov_normality_test(x_gauss_cauchy_noise)

#####kolmogorov-smirnov test

print("---------kolmogorov-smirnov test---------")
print('gauss')
x1_gauss = x_gauss[:n_x_gauss // 2]
x2_gauss = x_gauss[n_x_gauss // 2:]
KS_test(x1_gauss, x2_gauss)

print('gauss+uniform noise')
x1_uni_noise = x_gauss_uni_noise[:n_x_gauss // 2]
x2_uni_noise = x_gauss_uni_noise[n_x_gauss // 2:]
KS_test(x1_uni_noise, x2_uni_noise)

print('gauss+cauchy noise')
x1_cauchy_noise = x_gauss_cauchy_noise[:n_x_gauss // 2]
x2_cauchy_noise = x_gauss_cauchy_noise[n_x_gauss // 2:]
KS_test(x1_cauchy_noise, x2_cauchy_noise)