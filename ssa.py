import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from cycler import cycler
from ssa_module import plot_2d, X_to_TS, Hankelise, components_auto_group, SSA_simple_plt, SSA_plt_groups, SSA

plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 14
plt.rcParams['image.cmap'] = 'plasma'
plt.rcParams['axes.linewidth'] = 2

cols = plt.get_cmap('tab10').colors
plt.rcParams['axes.prop_cycle'] = cycler(color=cols)

"""# 1. Загрузка данных <a name="Section1"></a>"""

data = pd.read_csv("data_gulf_of_finland.csv") # берем ежедневные данные о температуре, собранные с 1 станции в ЛО
data = data.loc[data["NAME"] == "BELOGORKA, RS"][122:-2]
data = data.set_index("DATE")  # установим дату в качестве индекса
data.TEMP = (data.TEMP - 32) * 5 / 9  # преобразуем из градусов по Фаренгейту в градусы по Цельсию
print(data["TEMP"])  # всего у нас 2651 записей с  2012 по 2020 год

plt.plot(data.TEMP)
plt.xlabel("Date")
plt.ylabel("temp")
plt.show()

N = len(data)
L = round(N * 0.3)  # Длина окна - 30% от длины ряда
print(N, L)
K = N - L + 1  # Число колонок в траекторной матрице
temp_values = data.TEMP.values
# Создаем траекторную матрицу, вытянув соответствующие подпоследовательности из F и сложив их в виде столбцов.
X = np.column_stack([temp_values[i:i + L] for i in range(0, K)])
# Примечание: i+L выше дает нам до i+L-1, так как верхние границы массива numpy не включаются.

"""Визуализация траекторной матрицы:
1. Элементы на антидиагоналях равны
2. Значения чередуются с определенным шагом
"""

fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
ax = ax.matshow(X)
plt.xlabel("$L$-вектора задержек")
plt.ylabel("$K$-вектора задержек")
fig.colorbar(ax, fraction=0.025)
ax.colorbar.set_label("$F(t)$")
plt.title("Траекторная матрица", pad=15)
plt.show()

d = np.linalg.matrix_rank(X)

U, Sigma, V = np.linalg.svd(X)
V = V.T

# Вычислите элементарные матрицы X, сохранив их в многомерном массиве NumPy.
# Для этого требуется вычислить sigma_i * U_i * (V_i)^T для каждого i
X_elem = np.array([Sigma[i] * np.outer(U[:, i], V[:, i]) for i in range(0, d)])

# Быстрая проверка: сумма всех элементарных матриц в X_elm должна быть равна X, с точностью до определенного порога:
if not np.allclose(X, X_elem.sum(axis=0), atol=1e-10):
    print("WARNING: The sum of X's elementary matrices is not equal to X!")

"""Рассмотрим первые 12 элементарных матриц"""

n = min(12, d)
for i in range(n):
    plt.subplot(4, 4, i + 1)
    title = "$\mathbf{X}_{" + str(i) + "}$"
    plot_2d(X_elem[i], title)
plt.tight_layout()
plt.show()

sigma_sumsq = (Sigma ** 2).sum()
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].plot(Sigma ** 2 / sigma_sumsq * 100, lw=2.5)
ax[0].set_xlim(0, 11)
ax[0].set_title("Относительный вклад $\mathbf{X}_i$ в Траекторную матрицу")
ax[0].set_xlabel("$i$")
ax[0].set_ylabel("Вклад (%)")
ax[1].plot((Sigma ** 2).cumsum() / sigma_sumsq * 100, lw=2.5)
ax[1].set_xlim(0, 11)
ax[1].set_title("Суммарный вклад $\mathbf{X}_i$ в Траекторную матрицу")
ax[1].set_xlabel("$i$")
ax[1].set_ylabel("Вклад (%)")
plt.show()

# Ганкелизация, пока закомментим чтобы не тратить слишком много времени на выполнение
# n = min(d, 12)
# for j in range(0, n):
#     print("Ганкелизация", j, "/", n)
#     plt.subplot(4, 4, j + 1)
#     title = r"$\tilde{\mathbf{X}}_{" + str(j) + "}$"
#     plot_2d(Hankelise(X_elem[j]), title)
# plt.tight_layout()
# plt.show()

n = min(12, d)
fig = plt.subplot()
color_cycle = cycler(color=plt.get_cmap('tab20').colors)
fig.axes.set_prop_cycle(color_cycle)

# Convert elementary matrices straight to a time series - no need to construct any Hankel matrices.
for i in range(n):
    F_i = X_to_TS(X_elem[i])
    fig.axes.plot(F_i, lw=2)

fig.axes.plot(data.TEMP, alpha=1, lw=1)
fig.set_xlabel("$t$")
fig.set_ylabel(r"$\tilde{F}_i(t)$")
legend = [r"$\tilde{F}_{%s}$" % i for i in range(n)] + ["$F$"]
fig.set_title("Первые 12 компонент исходного ряда")
fig.legend(legend, loc=(1.05, 0.1))
plt.show()

# Сначала получите веса w, так как они будут часто использоваться повторно.
# Примечание: список(np.arange(L)+1) возвращает последовательность от 1 до L (первая строка в определении w),
# [L]*(K-L-1) повторяет L K-L-1 раз (вторая строка в определении w)
# список(np.arange(L)+1)[::-1] отменяет первый список (эквивалентно третьей строке)
# Сложите все списки вместе, и у нас будет наш массив весов.
w = np.array(list(np.arange(L) + 1) + [L] * (K - L - 1) + list(np.arange(L) + 1)[::-1])

# Получите все компоненты серии игрушек, сохраните их в виде столбцов в массиве F_elem.
F_elem = np.array([X_to_TS(X_elem[i]) for i in range(d)])

# Вычислите индивидуальные взвешенные нормы, ||F_i||_w, сначала,
# затем возьмите обратный квадратный корень, чтобы нам не пришлось делать это позже.
F_wnorms = np.array([w.dot(F_elem[i] ** 2) for i in range(d)])
F_wnorms = F_wnorms ** -0.5

# Вычислите матрицу w-corr. Диагональные элементы равны 1, поэтому мы можем начать с матрицы идентичности
# и повторите все пары i и j (i != j), отметив, что Wij = Wji.
Wcorr = np.identity(d)
for i in range(d):
    for j in range(i + 1, d):
        Wcorr[i, j] = abs(w.dot(F_elem[i] * F_elem[j]) * F_wnorms[i] * F_wnorms[j])
        Wcorr[j, i] = Wcorr[i, j]

ax = plt.imshow(Wcorr)
plt.xlabel(r"$\tilde{F}_i$")
plt.ylabel(r"$\tilde{F}_j$")
plt.colorbar(ax.colorbar, fraction=0.045)
ax.colorbar.set_label("$W_{ij}$")
plt.clim(0, 1)
plt.title("Матрица W-корреляций временного ряда")
plt.show()

ax = plt.imshow(Wcorr)
plt.xlabel(r"$\tilde{F}_i$")
plt.ylabel(r"$\tilde{F}_j$")
plt.colorbar(ax.colorbar, fraction=0.045)
ax.colorbar.set_label("$W_{ij}$")
plt.xlim(-0.5, 6.5)
plt.ylim(6.5, -0.5)
plt.clim(0, 1)
plt.title(r"Матрица корелляций для компонентов с индексом 0–6")
plt.show()

# Группируем элементы между собой
result_groups = components_auto_group(Wcorr)
F_trend = X_to_TS(X_elem[result_groups[0]].sum(axis=0))
F_periodic1 = X_to_TS(X_elem[result_groups[1]].sum(axis=0))

plt.plot(data.TEMP, lw=1)
plt.plot(F_trend)
plt.plot(F_periodic1)

if len(result_groups) > 3:
    F_periodic2 = X_to_TS(X_elem[result_groups[2]].sum(axis=0))
    F_noise = X_to_TS(X_elem[result_groups[3][0]:].sum(axis=0))
    groups = ["тренд", "циклика 1", "циклика 2", "шум"]
    plt.plot(F_periodic2)
else:
    F_noise = X_to_TS(X_elem[result_groups[2][0]:].sum(axis=0))
    groups = ["тренд", "циклика 1", "шум"]

plt.plot(F_noise, alpha=0.5)
plt.xlabel("$t$")
plt.ylabel(r"$\tilde{F}^{(j)}$")
legend = ["$F$"] + [r"$\tilde{F}^{(\mathrm{%s})}$" % group for group in groups]
plt.legend(legend)
plt.title("Сгрупированные компоненты")
plt.show()

"""## 5. Длина окна<a name="Section5"></a>

### 5.1 $L = 2$ <a name="Section5.1"></a>
Длина окна 2 может показаться бесполезным выбором но показательным.
"""

SSA_simple_plt(data.TEMP, window_size=2)
SSA_simple_plt(data.TEMP, window_size=5)
SSA_plt_groups(data.TEMP, window_size=30)
SSA_plt_groups(data.TEMP, window_size=60)
SSA_plt_groups(data.TEMP, window_size=90)
SSA_plt_groups(data.TEMP, window_size=180)
SSA_plt_groups(data.TEMP, window_size=365)
SSA_plt_groups(data.TEMP, window_size=550)
SSA_plt_groups(data.TEMP, window_size=730)

# попробуем окно на 365, сгруппировав самостоятельно
F_ssa_L365 = SSA(data.TEMP, 365)
F_ssa_L365.reconstruct([0, 1]).plot()
F_ssa_L365.reconstruct([2]).plot()
F_ssa_L365.reconstruct([3, 4]).plot()
F_ssa_L365.reconstruct(slice(5, 365)).plot(alpha=0.7)
plt.plot(data.TEMP, lw=1)
plt.title("Сгруппированные (вручную) компоненты, $L=365")
plt.xlabel("$t$")
plt.ylabel(r"$\tilde{F}_i(t)$")
groups = ["тренд", "циклика 1", "циклика 2", "шум"]
legend = ["$F$"] + [r"$\tilde{F}^{(\mathrm{%s})}$" % group for group in groups]
plt.legend(legend)
plt.show()
