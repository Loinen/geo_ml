

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import FactorAnalysis, PCA
import tensorly as tl
from tensorly import unfold as tl_unfold
from tensorly.decomposition import parafac
# from scipy.linalg import khatri_rao
from tensorly.tenalg import khatri_rao
from tensor_utils import *

"""### Пример Тензора"""

time_factor = np.load("data/time_factor.npy")
neuron_factor = np.load("data/neuron_factor.npy")
trial_factor = np.load("data/trial_factor.npy")
latent = np.load("data/latent.npy")
observed = np.load("data/observed.npy")


factors_actual = (normalize(time_factor), normalize(neuron_factor), normalize(trial_factor))


X, rank = observed, 3

# TensorLy
weights, factors_tl = parafac(X, rank=rank)
print(weights, factors_tl)

# Восстановим исходный набор данных
M_tl = reconstruct(factors_tl)

# Определяем ошибку восстановлеия
rec_error_tl = np.mean((X-M_tl)**2)

# Визуализируем факторы полученные с помощью Tensorly
plot_factors(factors_tl, d=3)
plt.suptitle("Factors computed with TensorLy", y=1.1, fontsize=20);


def decompose_three_way(tensor, rank, max_iter=501, verbose=False):

    # a = np.random.random((rank, tensor.shape[0]))
    b = np.random.random((rank, tensor.shape[1]))
    c = np.random.random((rank, tensor.shape[2]))

    for epoch in range(max_iter):
        # optimize a
        input_a = khatri_rao([b.T, c.T])
        target_a = tl.unfold(tensor, mode=0).T
        a = np.linalg.solve(input_a.T.dot(input_a), input_a.T.dot(target_a))

        # optimize b
        input_b = khatri_rao([a.T, c.T])
        target_b = tl.unfold(tensor, mode=1).T
        b = np.linalg.solve(input_b.T.dot(input_b), input_b.T.dot(target_b))
                            
        # optimize c
        input_c = khatri_rao([a.T, b.T])
        target_c = tl.unfold(tensor, mode=2).T
        c = np.linalg.solve(input_c.T.dot(input_c), input_c.T.dot(target_c))

        if verbose and epoch % int(max_iter * .2) == 0:
            res_a = np.square(input_a.dot(a) - target_a)
            res_b = np.square(input_b.dot(b) - target_b)
            res_c = np.square(input_c.dot(c) - target_c)
            print("Epoch:", epoch, "| Loss (C):", res_a.mean(), "| Loss (B):", res_b.mean(), "| Loss (C):", res_c.mean())
        
    return a.T, b.T, c.T

factors_np = decompose_three_way(X, rank, verbose=False)

a, b, c = factors_np
factors = (normalize(a), normalize(b), normalize(c))

M_np = reconstruct(factors_np)
rec_error_np = np.mean((X-M_np)**2)

"""Сравнение между тремя подходами при оценке $X$."""

fig, axes = plt.subplots(1, 4, sharey=True)
tensors = [X, M_tl,  M_np]
titles = ["Ground Truth", "TensorLy", "Numpy"]
trial_num = 50

for title, tensor, ax in zip(titles, tensors, axes):
    ax.imshow(tensor[:, :, trial_num].T, cmap='bwr', aspect=20)
    ax.set_xlabel("Time")
    ax.set_title(title)
axes[0].set_ylabel("Neuron");

plot_factors(factors_np, d=3)

# a[:, 1] = a[:, 1] * -1
# b[:, 1] = b[:, 1] * -1
# c[:, 2] = c[:, 2] * -1

"""Нормализуем факторы, для того чтобы сравнить с ground true"""

factors = (normalize(a), normalize(b), normalize(c))

fig, axes = plt.subplots(rank, 3, figsize=(8, int(rank * 1.2 + 1)))
compare_factors(factors, factors_actual, factors_ind=[1, 0, 2], fig=fig);

"""### Сравнение трех подходов

#### Ошибка восстановления
"""

import timeit

iter_num = 50
times = {'tl': [], 'tt': [], 'np': []}
rec_errors = {'tl': [], 'tt': [], 'np': []}

# TensorLy
for i in range(iter_num):
    start_time = timeit.default_timer()
    weights, factors_tl = parafac(X, rank=rank, n_iter_max=200)
    end_time = timeit.default_timer() - start_time
    times['tl'].append(end_time)
    M_tl = reconstruct(factors_tl)
    rec_error_tl = np.mean((X-M_tl)**2)
    rec_errors['tl'].append(rec_error_tl)
    print("TensorLy | Iteration: {} / {} | time take: {} sec".format(i+1, iter_num, end_time))

# Numpy
for i in range(iter_num):
    start_time = timeit.default_timer()
    factors_np = decompose_three_way(X, rank, max_iter=200, verbose=False)
    end_time = timeit.default_timer() - start_time
    times['np'].append(end_time)
    M_np = reconstruct(factors_np)
    rec_error_np = np.mean((X-M_np)**2)
    rec_errors['np'].append(rec_error_np)
    print("Numpy | Iteration: {} / {} | time take: {} sec".format(i+1, iter_num, end_time))

plt.figure(figsize=(5, 5)) #, dpi=200)
factor = 1e15
plt.scatter(np.array(times['tl']), np.array(rec_errors['tl']), c='red', label="TensorLy")
plt.scatter(np.array(times['tt']), np.array(rec_errors['tt']), c='green', label="tensortools")
plt.scatter(np.array(times['np']), np.array(rec_errors['np']), c='b', label="Numpy")
plt.xlabel("Execution Time (sec)", fontsize=25)
plt.ylabel("Reconstruction Error", fontsize=25)
plt.ylim(0.85 * 1e-15, 1.17 * 1e-15)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(loc="lower left", fontsize=13);
plt.grid()

# plt.savefig("metric-1.png", transparent=True, bbox_inches='tight')

"""**Обратите внимание**, что в зависимости от результата этого теста некоторые значения могут быть отклонениями. Ниже приведен порог, который я применил для удаления выбросов."""

lim = 3 * 1e-15
pkg = 'tt'
times[pkg] = np.array(times[pkg])[np.array(rec_errors[pkg]) < lim].tolist()
rec_errors[pkg] = np.array(rec_errors[pkg])[np.array(rec_errors[pkg]) < lim].tolist()