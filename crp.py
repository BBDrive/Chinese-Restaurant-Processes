# from https://github.com/roboticcam/python_machine_learning/blob/master/chinese_restaurant_process.ipynb
import numpy as np
from scipy.special import digamma, gamma, gammaln
from math import log, exp
'''
中国餐馆过程（Chinese Restaurant Process）
这是用递归的方式写的第一种Stirling值的取得方法
s(n+1, k) = ns(n, k) + s(n, k-1)
be aware that this can be quite slow when n is large
'''


def stirling(n, k):
    if n <= 0:
        return 1
    elif k <= 0:
        return 0
    elif n == k:
        return 1
    elif n != 0 and n == k:
        return 1
    elif k > n:
        return 0
    else:
        return (n - 1) * stirling(n - 1, k) + stirling(n - 1, k - 1)


# direct sampling
def Draw_CRP_Direct_Sample(N=10, alpha=3, T=50):
    Z_table = np.zeros((T, N))
    for t in range(T):
        Z = np.zeros(N, dtype=int)
        for i in range(N):
            if i == 0:
                Z[i] = 1
            else:
                unique, counts = np.unique(Z, return_counts=True)

                # remove the zeros unsigned tables
                if unique[0] == 0:
                    unique = np.delete(unique, 0)
                    counts = np.delete(counts, 0)

                # added alpha to the end of the counts (weights) array
                counts = np.append(counts, alpha)

                # also the new table index will be the max of table seen so far
                unique = np.append(unique, max(unique) + 1)

                u = np.random.uniform() * sum(counts)

                a_counts = np.cumsum(counts)

                # first index where accumuated sum is greater than random variable
                index = np.argmax(a_counts > u)

                Z[i] = unique[index]

        Z_table[t, :] = Z
    return Z_table


# Gibbs sampling
def Draw_CRP_Gibbs_Sample(N=10, alpha=3, T=50, burn_in=10):
    Z = np.ones(N, dtype=int)
    Z_table = np.zeros((T, N))

    for t in range(T + burn_in):
        for i in range(N):
            # remove current table assignment
            Z[i] = 0

            unique, counts = np.unique(Z, return_counts=True)

            # remove the zeros in unassigned tables
            if unique[0] == 0:
                unique = np.delete(unique, 0)
                counts = np.delete(counts, 0)

            # added alpha to the end of the counts (weights) array
            counts = np.append(counts, alpha)

            # also the new table index will be the max of table seen so far
            unique = np.append(unique, max(unique) + 1)

            u = np.random.uniform() * sum(counts)

            a_counts = np.cumsum(counts)

            # first index where accumuated sum is greater than random variable
            index = np.argmax(a_counts > u)

            Z[i] = unique[index]

        old_table = np.unique(Z)
        new_table = np.array(range(1, len(old_table) + 1))

        for k in range(len(old_table)):
            Z[Z == old_table[k]] = new_table[k]

        if t >= burn_in:
            Z_table[t - burn_in, :] = Z

    return Z_table

'''
in the following, we compare sample $\widehat{E}(K)$ vs true $E(K)$, and sample $\widehat{P}(K = k)$ vs true$P(K= k)$
$K$ is the number of tables

在以下的函数里，我们对$K$的样本期望值和$K$理论期望值进行比较; 我们还对$P(K=k)$的样本值和$P(K=k)$理论期望值进行比较;
'''

alpha = 1
N = 5
T = 100
burn_in = int(T / 10)

# ------------------------------------------
# call either of the following function
# 您可以选择运行下面的其中一个函数
# ------------------------------------------

# Z_table = Draw_CRP_Direct_Sample(N, alpha, T)

Z_table = Draw_CRP_Gibbs_Sample(N, alpha, T, burn_in)

table_numbers = np.zeros(T, dtype=int)

for t in range(T):
    unique, count = np.unique(Z_table[t, :], return_counts=True)
    table_numbers[t] = len(count)

# ------------------------------------------
# we compare sample E(K) vs true E(K)
# 我们对 K 的样本期望值和理论期望值进行比较
# ------------------------------------------

exp_average = np.mean(table_numbers)
theory_average = alpha * (digamma(alpha + N) - digamma(alpha))
print("sample E(K) = ", exp_average, " theorical E(K) = ", theory_average)

# ------------------------------------------
# We compared Pr(K = k) between sample and true
# 我们对 Pr(K=k) 的样本期望值和理论期望值进行比较
# ------------------------------------------

unique, count = np.unique(table_numbers, return_counts=True)

for t in range(len(unique)):
    k = unique[t]
    exp_prob = count[t] * 1.000 / T

    # ------------------------------------------
    # to avoid overflow, we use a little trick:
    # 为防止变量接近0，或变成极值，我们会用以下的小技巧:
    # a*b/c = exp(log(a)+log(b)-log(c))
    # ------------------------------------------

    theory_prob = log(stirling(N, k)) + k * log(alpha) + gammaln(alpha) - gammaln(alpha + N)
    theory_prob = exp(theory_prob)

    print("sample Pr(K = ", unique[t], ") = ", exp_prob, "; theorical Pr(K = ", unique[t], ") = ", theory_prob)