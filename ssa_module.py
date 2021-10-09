import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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

    HX = np.zeros((L, K))

    for m in range(L):
        for n in range(K):
            s = m + n
            if 0 <= s <= L - 1:
                for l in range(0, s + 1):
                    HX[m, n] += 1 / (s + 1) * X[l, s - l]
            elif L <= s <= K - 1:
                for l in range(0, L - 1):
                    HX[m, n] += 1 / (L - 1) * X[l, s - l]
            elif K <= s <= K + L - 2:
                for l in range(s - K + 1, L):
                    HX[m, n] += 1 / (K + L - s - 1) * X[l, s - l]
    if transpose:
        return HX.T
    else:
        return HX


def plot_2d(m, title=""):
    plt.imshow(m)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)


def X_to_TS(X_i):
    """Усредняет антидиагонали данной элементарной матрицы X_i и возвращает временной ряд."""
    # Изменить порядок столбцов X_i
    X_rev = X_i[::-1]
    return np.array([X_rev.diagonal(i).mean() for i in range(-X_i.shape[0] + 1, X_i.shape[1])])


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

        Note -  Even if an NumPy array or list is used for the initial time series, all time series returned will be
        in the form of a Pandas Series or DataFrame object.
        """

        # Tedious type-checking for the initial time series
        if not isinstance(tseries, self.__supported_types):
            raise TypeError("Unsupported time series object. Try Pandas Series, NumPy array or list.")

        # Checks to save us from ourselves
        self.N = len(tseries)
        if not 2 <= L <= self.N / 2:
            raise ValueError("The window length must be in the interval [2, N/2].")

        self.L = L
        self.orig_TS = pd.Series(tseries)
        self.K = self.N - self.L + 1

        # Embed the time series in a trajectory matrix
        self.X = np.array([self.orig_TS.values[i:L + i] for i in range(0, self.K)]).T

        # Decompose the trajectory matrix
        self.U, self.Sigma, VT = np.linalg.svd(self.X)
        self.d = np.linalg.matrix_rank(self.X)

        self.TS_comps = np.zeros((self.N, self.d))

        if not save_mem:
            # Construct and save all the elementary matrices
            self.X_elem = np.array([self.Sigma[i] * np.outer(self.U[:, i], VT[i, :]) for i in range(self.d)])

            # Diagonally average the elementary matrices, store them as columns in array.
            for i in range(self.d):
                X_rev = self.X_elem[i, ::-1]
                self.TS_comps[:, i] = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0] + 1, X_rev.shape[1])]

            self.V = VT.T
        else:
            # Reconstruct the elementary matrices without storing them
            for i in range(self.d):
                X_elem = self.Sigma[i] * np.outer(self.U[:, i], VT[i, :])
                X_rev = X_elem[::-1]
                self.TS_comps[:, i] = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0] + 1, X_rev.shape[1])]

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

        ts_vals = self.TS_comps[:, indices].sum(axis=1)
        return pd.Series(ts_vals, index=self.orig_TS.index)

    def calc_wcorr(self):
        """
        Calculates the w-correlation matrix for the time series.
        """

        # Calculate the weights
        w = np.array(list(np.arange(self.L) + 1) + [self.L] * (self.K - self.L - 1) + list(np.arange(self.L) + 1)[::-1])

        def w_inner(F_i, F_j):
            return w.dot(F_i * F_j)

        # Calculated weighted norms, ||F_i||_w, then invert.
        F_wnorms = np.array([w_inner(self.TS_comps[:, i], self.TS_comps[:, i]) for i in range(self.d)])
        F_wnorms = F_wnorms ** -0.5

        # Calculate Wcorr.
        self.Wcorr = np.identity(self.d)
        for i in range(self.d):
            for j in range(i + 1, self.d):
                self.Wcorr[i, j] = abs(w_inner(self.TS_comps[:, i], self.TS_comps[:, j]) * F_wnorms[i] * F_wnorms[j])
                self.Wcorr[j, i] = self.Wcorr[i, j]

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
        plt.clim(0, 1)

        # For plotting purposes:
        if max == self.d:
            max_rnge = self.d - 1
        else:
            max_rnge = max

        plt.xlim(min - 0.5, max_rnge + 0.5)
        plt.ylim(max_rnge + 0.5, min - 0.5)

        return self.Wcorr


def components_auto_group(wcorr_matrix, n_elem=7):
    groups_list = []
    temp_list = []
    result_groups = []
    check_elem = 0

    for inx, mstr in enumerate(wcorr_matrix[:n_elem]):
        for x in range(n_elem):
            if x != inx and mstr[x] > 0.4:
                if [inx, x] not in groups_list:
                    groups_list.append([x, inx])
    print(groups_list)
    while groups_list[0][1] != check_elem:
        result_groups.append([check_elem])
        check_elem += 1
    temp_list.append(check_elem)
    for x in range(0, len(groups_list)):
        if groups_list[x][1] == check_elem:
            if groups_list[x][0] not in temp_list:
                temp_list.append(groups_list[x][0])
        elif groups_list[x][1] == check_elem + 1 and groups_list[x][1] in temp_list or groups_list[x][0] in temp_list:
            if groups_list[x][0] not in temp_list:
                temp_list.append(groups_list[x][0])
        else:
            if len(temp_list) > 0:
                result_groups.append(temp_list)
            temp_list = [groups_list[x][1], groups_list[x][0]]
        check_elem = groups_list[x][1]
    if len(temp_list) > 0:
        result_groups.append(temp_list)
    print(result_groups)
    return result_groups


def SSA_simple_plt(data, window_size):
    F_ssa_L2 = SSA(data, window_size)
    F_ssa_L2.components_to_df().plot()
    F_ssa_L2.orig_TS.plot(alpha=0.4)
    plt.xlabel("$t$")
    plt.ylabel(r"$\tilde{F}_i(t)$")
    plt.title(r"$L={%d}$" % window_size)
    plt.show()


def SSA_plt_groups(data, window_size):
    print("window_size", window_size)
    F_ssa_L20 = SSA(data, window_size)
    wcorr = F_ssa_L20.plot_wcorr()
    plt.title("W-Correlation, $L={%d}$" % window_size)
    plt.show()
    if window_size > 10:
        F_ssa_L20.plot_wcorr(max=10)
        plt.title("W-Correlation, $L={%d}$ (first 10)" % window_size)
        plt.show()

    result_groups = components_auto_group(wcorr)
    groups = ["тренд", "шум"]

    if len(result_groups) == 1:
        F_ssa_L20.reconstruct(result_groups[0][0]).plot()  # trend
        F_ssa_L20.reconstruct(slice(result_groups[0][1], window_size)).plot(alpha=0.5)
    else:
        F_ssa_L20.reconstruct(result_groups[0]).plot()  # trend

    if len(result_groups) > 3:
        F_ssa_L20.reconstruct(result_groups[1]).plot()  # periodic
        F_ssa_L20.reconstruct(result_groups[2]).plot()  # periodic 2
        F_ssa_L20.reconstruct(slice(result_groups[3][0], window_size)).plot(alpha=0.5)
        groups = ["тренд", "циклика1", "циклика2", "шум"]
    elif len(result_groups) == 2:
        F_ssa_L20.reconstruct(slice(result_groups[1][0], window_size)).plot(alpha=0.5)
    elif len(result_groups) == 3:
        F_ssa_L20.reconstruct(result_groups[1]).plot()  # periodic
        F_ssa_L20.reconstruct(slice(result_groups[2][0], window_size)).plot(alpha=0.5)
        groups = ["тренд", "циклика", "шум"]

    # F_ssa_L20.reconstruct(3).plot()
    plt.xlabel("$t$")
    plt.ylabel(r"$\tilde{F}_i(t)$")
    plt.title("Группировка компонент, $L={%d}$" % window_size)
    legend = [r"$\tilde{F}^{(\mathrm{%s})}$" % group for group in groups]
    plt.legend(legend)
    plt.show()

    if window_size > 10:
        F_ssa_L20.components_to_df(n=7).plot()
        plt.title(r"Первые 7 компонент, $L={%d}$" % window_size)
        plt.xlabel(r"$t$")
        plt.show()
