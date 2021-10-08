import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from cycler import cycler

plt.rcParams['figure.figsize'] = (10,8)
plt.rcParams['font.size'] = 14
plt.rcParams['image.cmap'] = 'plasma'
plt.rcParams['axes.linewidth'] = 2

cols = plt.get_cmap('tab10').colors
plt.rcParams['axes.prop_cycle'] = cycler(color=cols)


def plot_2d(m, title=""):
    plt.imshow(m)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.show()

"""# 1. Загрузка данных <a name="Section1"></a>"""

data = pd.read_csv("data_gulf_of_finland.csv")
data = data.loc[data["NAME"] == "BELOGORKA, RS"] # берем ежедневные данные о температуре, собранные с 1 станции в ЛО
data = data.set_index("DATE") # установим дату в качестве индекса
data.TEMP = (data.TEMP - 32) * 5/9 # преобразуем из градусов по Фаренгейту в градусы по Цельсию
data["TEMP"] # всего у нас 2651 записей с  2012 по 2020 год

plt.plot(data.TEMP)
plt.xlabel("Date")
plt.ylabel("temp")
plt.show()


N = len(data)
L = round(N * 0.3) # Длина окна - 30% от длины ряда
print(N, L)
K = N - L + 1 # Число колонок в траекторной матрице
# Создаем траекторную матрицу, вытянув соответствующие подпоследовательности из F и сложив их в виде столбцов.
X = np.column_stack([data.TEMP.values[i:i+L] for i in range(0,K)])
# Примечание: i+L выше дает нам до i+L-1, так как верхние границы массива numpy не включаются.

"""Визуализация траекторной матрицы:
1. Элементы на антидиагоналях равны
2. Значения чередуются с определенным шагом
"""

fig, ax = plt.subplots(figsize=(8,4), dpi=150)
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

# Вычислите элементарные матрицы X, сохранив их в многомерном массиве NumPy. Для этого требуется вычислить sigma_i * U_i * (V_i)^T для каждого i 
X_elem = np.array([Sigma[i] * np.outer(U[:,i], V[:,i]) for i in range(0,d)])

# # Быстрая проверка: сумма всех элементарных матриц в X_elm должна быть равна X, с точностью до определенного порога:
# if not np.allclose(X, X_elem.sum(axis=0), atol=1e-10):
#     print("WARNING: The sum of X's elementary matrices is not equal to X!")

"""Рассмотрим первые 12 элементарных матриц"""

n = min(12, d) 
for i in range(n):
    plt.subplot(4,4,i+1)
    title = "$\mathbf{X}_{" + str(i) + "}$"
    plot_2d(X_elem[i], title)
plt.tight_layout()
plt.show()

sigma_sumsq = (Sigma**2).sum()
fig, ax = plt.subplots(1, 2, figsize=(14,5))
ax[0].plot(Sigma**2 / sigma_sumsq * 100, lw=2.5)
ax[0].set_xlim(0,11)
ax[0].set_title("Относительный вклад $\mathbf{X}_i$ в Траекторную матрицу")
ax[0].set_xlabel("$i$")
ax[0].set_ylabel("Вклад (%)")
ax[1].plot((Sigma**2).cumsum() / sigma_sumsq * 100, lw=2.5)
ax[1].set_xlim(0,11)
ax[1].set_title("Суммарный вклад $\mathbf{X}_i$ в Траекторную матрицу")
ax[1].set_xlabel("$i$")
ax[1].set_ylabel("Вклад (%)");
plt.show()


def Hankelise(X):
    L, K = X.shape
    transpose = False
    if L > K:
        # # Приведенная ниже Ганкелезиция работает только для матриц, где L < K.
        # # Чтобы изменить матрицу L > K, сначала поменяйте местами L и K и транспонируйте X.
        # Установите флаг для транспонирования HX перед возвратом.
        X = X.T
        L, K = K, L
        transpose = True

    HX = np.zeros((L,K))
    
    for m in range(L):
        for n in range(K):
            s = m+n
            if 0 <= s <= L-1:
                for l in range(0,s+1):
                    HX[m,n] += 1/(s+1)*X[l, s-l]    
            elif L <= s <= K-1:
                for l in range(0,L-1):
                    HX[m,n] += 1/(L-1)*X[l, s-l]
            elif K <= s <= K+L-2:
                for l in range(s-K+1,L):
                    HX[m,n] += 1/(K+L-s-1)*X[l, s-l]
    if transpose:
        return HX.T
    else:
        return HX

n = min(d, 12)
for j in range(0,n):
    plt.subplot(4,4,j+1)
    title = r"$\tilde{\mathbf{X}}_{" + str(j) + "}$"
    plot_2d(Hankelise(X_elem[j]), title)
plt.tight_layout()
plt.show()


def X_to_TS(X_i):
    """Усредняет антидиагонали данной элементарной матрицы X_i и возвращает временной ряд."""
    # Изменить порядок столбцов X_i
    X_rev = X_i[::-1]
    return np.array([X_rev.diagonal(i).mean() for i in range(-X_i.shape[0]+1, X_i.shape[1])])

n = min(12,d) 
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
legend = [r"$\tilde{F}_{%s}$" %i for i in range(n)] + ["$F$"]
fig.set_title("Первые 12 компонент исходного ряда")
fig.legend(legend, loc=(1.05,0.1));
plt.show()



# Группируем элементы между собой
F_trend = X_to_TS(X_elem[[0,1,6]].sum(axis=0))
F_periodic1 = X_to_TS(X_elem[[2,3]].sum(axis=0))
F_periodic2 = X_to_TS(X_elem[[4,5]].sum(axis=0))
F_noise = X_to_TS(X_elem[7:].sum(axis=0))

plt.plot(data.TEMP, lw=1)
plt.plot(F_trend)
plt.plot(F_periodic1)
plt.plot(F_periodic2)
plt.plot(F_noise, alpha=0.5)
plt.xlabel("$t$")
plt.ylabel(r"$\tilde{F}^{(j)}$")
groups = ["тренд", "циклика 1", "циклика 2", "шум"]
legend = ["$F$"] + [r"$\tilde{F}^{(\mathrm{%s})}$"%group for group in groups]
plt.legend(legend)
plt.title("Сгрупированные компоненты")
plt.show()

# # A list of tuples so we can create the next plot with a loop.
# components = [("Тренд", trend, F_trend),
#               ("циклика 1", periodic1, F_periodic1),
#               ("циклика 2", periodic2, F_periodic2),
#               ("шум", noise, F_noise)]
#
# # Plot the separated components and original components together.
# fig = plt.figure()
# n=1
# for name, orig_comp, ssa_comp in components:
#     ax = fig.add_subplot(2,2,n)
#     ax.plot(t, orig_comp, linestyle="--", lw=2.5, alpha=0.7)
#     ax.plot(t, ssa_comp)
#     ax.set_title(name, fontsize=16)
#     ax.set_xticks([])
#     n += 1

fig.tight_layout()
plt.show()


# Сначала получите веса w, так как они будут часто использоваться повторно.
# Примечание: список(np.arange(L)+1) возвращает последовательность от 1 до L (первая строка в определении w),
# [L]*(K-L-1) повторяет L K-L-1 раз (вторая строка в определении w)
# список(np.arange(L)+1)[::-1] отменяет первый список (эквивалентно третьей строке)
# Сложите все списки вместе, и у нас будет наш массив весов.
w = np.array(list(np.arange(L)+1) + [L]*(K-L-1) + list(np.arange(L)+1)[::-1])

# Получите все компоненты серии игрушек, сохраните их в виде столбцов в массиве F_elem.
F_elem = np.array([X_to_TS(X_elem[i]) for i in range(d)])

# Вычислите индивидуальные взвешенные нормы, ||F_i||_w, сначала, затем возьмите обратный квадратный корень, чтобы нам не пришлось делать это позже.
F_wnorms = np.array([w.dot(F_elem[i]**2) for i in range(d)])
F_wnorms = F_wnorms**-0.5

# Вычислите матрицу w-corr. Диагональные элементы равны 1, поэтому мы можем начать с матрицы идентичности
# и повторите все пары i и j (i != j), отметив, что Wij = Wji.
Wcorr = np.identity(d)
for i in range(d):
    for j in range(i+1,d):
        Wcorr[i,j] = abs(w.dot(F_elem[i]*F_elem[j]) * F_wnorms[i] * F_wnorms[j])
        Wcorr[j,i] = Wcorr[i,j]

ax = plt.imshow(Wcorr)
plt.xlabel(r"$\tilde{F}_i$")
plt.ylabel(r"$\tilde{F}_j$")
plt.colorbar(ax.colorbar, fraction=0.045)
ax.colorbar.set_label("$W_{ij}$")
plt.clim(0,1)
plt.title("Матрица W-корреляций временного ряда");
plt.show()

"""Структура $\mathbf{W}_{\text{corr}}$ показывает большую корреляцию между компонентами временных рядов, особенно в диапазоне $7 \le i,j \le 69$. Поскольку это были компоненты, которые мы классифицировали как принадлежащие к шуму во временном ряду, неудивительно, что между ними всеми существуют незначительные корреляции; это естественный результат того, что шум не имеет базового структурного компонента, который можно было бы дополнительно разделить.

Важно отметить, что $\mathbf{W}_{\text{corr}}$ примерно разделен на два "блока": $0 \le i,j \le 6$ и $7 \le i,j \le 69$. Это соответствует двум основным группам: сглаженный временной ряд (т. е. тренд плюс две периодические компоненты) и остаточный шум. Масштабирование первых семи компонентов в $\mathbf{W}_{\text{corr}}$:
"""

ax = plt.imshow(Wcorr)
plt.xlabel(r"$\tilde{F}_i$")
plt.ylabel(r"$\tilde{F}_j$")
plt.colorbar(ax.colorbar, fraction=0.045)
ax.colorbar.set_label("$W_{ij}$")
plt.xlim(-0.5,6.5)
plt.ylim(6.5,-0.5)
plt.clim(0,1)
plt.title(r"Матрица корелляций для компонентов с индексом 0–6");
plt.show()

class SSA(object):
    
    __supported_types = (pd.Series, np.ndarray, list)
    
    def __init__(self, tseries, L, save_mem=True):
        """
        Decomposes the given time series with a singular-spectrum analysis. Assumes the values of the time series are
        recorded at equal intervals.
        
        Parameters
        ----------
        tseries : The original time series, in the form of a Pandas Series, NumPy array or list. 
        L : The window length. Must be an integer 2 <= L <= N/2, where N is the length of the time series.
        save_mem : Conserve memory by not retaining the elementary matrices. Recommended for long time series with
            thousands of values. Defaults to True.
        
        Note: Even if an NumPy array or list is used for the initial time series, all time series returned will be
        in the form of a Pandas Series or DataFrame object.
        """
        
        # Tedious type-checking for the initial time series
        if not isinstance(tseries, self.__supported_types):
            raise TypeError("Unsupported time series object. Try Pandas Series, NumPy array or list.")
        
        # Checks to save us from ourselves
        self.N = len(tseries)
        if not 2 <= L <= self.N/2:
            raise ValueError("The window length must be in the interval [2, N/2].")
        
        self.L = L
        self.orig_TS = pd.Series(tseries)
        self.K = self.N - self.L + 1
        
        # Embed the time series in a trajectory matrix
        self.X = np.array([self.orig_TS.values[i:L+i] for i in range(0, self.K)]).T
        
        # Decompose the trajectory matrix
        self.U, self.Sigma, VT = np.linalg.svd(self.X)
        self.d = np.linalg.matrix_rank(self.X)
        
        self.TS_comps = np.zeros((self.N, self.d))
        
        if not save_mem:
            # Construct and save all the elementary matrices
            self.X_elem = np.array([ self.Sigma[i]*np.outer(self.U[:,i], VT[i,:]) for i in range(self.d) ])

            # Diagonally average the elementary matrices, store them as columns in array.           
            for i in range(self.d):
                X_rev = self.X_elem[i, ::-1]
                self.TS_comps[:,i] = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0]+1, X_rev.shape[1])]
            
            self.V = VT.T
        else:
            # Reconstruct the elementary matrices without storing them
            for i in range(self.d):
                X_elem = self.Sigma[i]*np.outer(self.U[:,i], VT[i,:])
                X_rev = X_elem[::-1]
                self.TS_comps[:,i] = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0]+1, X_rev.shape[1])]
            
            self.X_elem = "Re-run with save_mem=False to retain the elementary matrices."
            
            # The V array may also be very large under these circumstances, so we won't keep it.
            self.V = "Re-run with save_mem=False to retain the V matrix."
        
        # Calculate the w-correlation matrix.
        self.calc_wcorr()
            
    def components_to_df(self, n=0):
        """
        Returns all the time series components in a single Pandas DataFrame object.
        """
        if n > 0:
            n = min(n, self.d)
        else:
            n = self.d
        
        # Create list of columns - call them F0, F1, F2, ...
        cols = ["F{}".format(i) for i in range(n)]
        return pd.DataFrame(self.TS_comps[:, :n], columns=cols, index=self.orig_TS.index)
            
    
    def reconstruct(self, indices):
        """
        Reconstructs the time series from its elementary components, using the given indices. Returns a Pandas Series
        object with the reconstructed time series.
        
        Parameters
        ----------
        indices: An integer, list of integers or slice(n,m) object, representing the elementary components to sum.
        """
        if isinstance(indices, int): indices = [indices]
        
        ts_vals = self.TS_comps[:,indices].sum(axis=1)
        return pd.Series(ts_vals, index=self.orig_TS.index)
    
    def calc_wcorr(self):
        """
        Calculates the w-correlation matrix for the time series.
        """
             
        # Calculate the weights
        w = np.array(list(np.arange(self.L)+1) + [self.L]*(self.K-self.L-1) + list(np.arange(self.L)+1)[::-1])
        
        def w_inner(F_i, F_j):
            return w.dot(F_i*F_j)
        
        # Calculated weighted norms, ||F_i||_w, then invert.
        F_wnorms = np.array([w_inner(self.TS_comps[:,i], self.TS_comps[:,i]) for i in range(self.d)])
        F_wnorms = F_wnorms**-0.5
        
        # Calculate Wcorr.
        self.Wcorr = np.identity(self.d)
        for i in range(self.d):
            for j in range(i+1,self.d):
                self.Wcorr[i,j] = abs(w_inner(self.TS_comps[:,i], self.TS_comps[:,j]) * F_wnorms[i] * F_wnorms[j])
                self.Wcorr[j,i] = self.Wcorr[i,j]
    
    def plot_wcorr(self, min=None, max=None):
        """
        Plots the w-correlation matrix for the decomposed time series.
        """
        if min is None:
            min = 0
        if max is None:
            max = self.d
        
        if self.Wcorr is None:
            self.calc_wcorr()
        
        ax = plt.imshow(self.Wcorr)
        plt.xlabel(r"$\tilde{F}_i$")
        plt.ylabel(r"$\tilde{F}_j$")
        plt.colorbar(ax.colorbar, fraction=0.045)
        ax.colorbar.set_label("$W_{i,j}$")
        plt.clim(0,1)
        
        # For plotting purposes:
        if max == self.d:
            max_rnge = self.d-1
        else:
            max_rnge = max
        
        plt.xlim(min-0.5, max_rnge+0.5)
        plt.ylim(max_rnge+0.5, min-0.5)

"""## 5. Длина окна<a name="Section5"></a>

### 5.1 $L = 2$ <a name="Section5.1"></a>
Длина окна 2 может показаться бесполезным выбором но показательным.
"""

F_ssa_L2 = SSA(data.TEMP, 2)
F_ssa_L2.components_to_df().plot()
F_ssa_L2.orig_TS.plot(alpha=0.4)
plt.xlabel("$t$")
plt.ylabel(r"$\tilde{F}_i(t)$")
plt.title(r"$L=2$ ");
plt.show()


F_ssa_L5 = SSA(data.TEMP, 5)
F_ssa_L5.components_to_df().plot()
F_ssa_L5.orig_TS.plot(alpha=0.4)
plt.xlabel("$t$")
plt.ylabel(r"$\tilde{F}_i(t)$")
plt.title(r"$L=5$");
plt.show()



F_ssa_L20 = SSA(data.TEMP, 20)
F_ssa_L20.plot_wcorr()
plt.title("W-Correlation, $L=20$");
plt.show()


F_ssa_L20.reconstruct(0).plot()
F_ssa_L20.reconstruct([1,2,3]).plot()
F_ssa_L20.reconstruct(slice(4,20)).plot()
F_ssa_L20.reconstruct(3).plot()
plt.xlabel("$t$")
plt.ylabel(r"$\tilde{F}_i(t)$")
plt.title("Группировка компонент, $L=20$");
plt.legend([r"$\tilde{F}_0$", 
            r"$\tilde{F}_1+\tilde{F}_2+\tilde{F}_3$", 
            r"$\tilde{F}_4+ \ldots + \tilde{F}_{19}$",
            r"$\tilde{F}_3$"]);
plt.show()

F_ssa_L40 = SSA(data.TEMP, 40)
F_ssa_L40.plot_wcorr()
plt.title("Матрица корелляций, $L=40$");
plt.show()


F_ssa_L40.reconstruct(0).plot()
F_ssa_L40.reconstruct([1,2,3]).plot()
F_ssa_L40.reconstruct([4,5]).plot()
F_ssa_L40.reconstruct(slice(6,40)).plot(alpha=0.7)
plt.title("Сгруппированные компоненты, $L=40$")
plt.xlabel("$t$")
plt.ylabel(r"$\tilde{F}_i(t)$")
plt.legend([r"$\tilde{{F}}^{{({0})}}$".format(i) for i in range(4)]);
plt.show()


F_ssa_L60 = SSA(data.TEMP, 60)
F_ssa_L60.plot_wcorr()
plt.title("Матрица корреляций, $L=60$");
plt.show()

"""Как и в исходном результате $L=70$, матрица w-корреляции теперь состоит из двух отдельных блоков: $\tilde{F}_0$ до $\tilde{F}_6$ и $\tilde{F}_7$ до $\tilde{F}_{59}$. Из опыта теперь ясно, что $\tilde{F}^{\text{(signal)}} = \sum {i=0}^6 \tilde{F}_i$ будет комбинированной тенденцией и периодическими компонентами ("сигнал"), а $\tilde{F}^{\text{(noise)}} = \sum {i=7}^{59} \tilde{F}_i$ будет шумом:"""

F_ssa_L60.reconstruct(slice(0,7)).plot()
F_ssa_L60.reconstruct(slice(7,60)).plot()
plt.legend([r"$\tilde{F}^{\mathrm{(signal)}}$", r"$\tilde{F}^{\mathrm{(noise)}}$"])
plt.title("Сигнал и шум исходного ряда, $L = 60$")
plt.xlabel(r"$t$");
plt.show()

F_ssa_L60.plot_wcorr(max=6)
plt.title("Матрица корреляции, $L=60$");
plt.show()

"""Чтобы понять, почему существует незначительная w-корреляция между большинством из первых семи компонентов, будет разумно построить их все сразу:"""

F_ssa_L60.components_to_df(n=7).plot()
plt.title(r"Первые 7 компонент, $L=60$")
plt.xlabel(r"$t$");
plt.show()

