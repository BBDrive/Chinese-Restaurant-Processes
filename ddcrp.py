# Distance dependent Chinese restaurant processes
import numpy as np
import multiprocessing as mp
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data[:]

# alpha增大， a减小，新桌子增多
def ddcrp(customer: np.array, alpha=5, a=3, step=100):
    customer_num = customer.shape[0]
    D = np.zeros((customer_num, customer_num), dtype=float)
    table_num = 0
    for i, data in enumerate(customer):
        # exponential decay
        prob = np.linalg.norm(data - customer, ord=2, axis=1)
        prob = np.exp(-prob/a)
        prob[i] = alpha
        prob = prob/prob.sum()
        D[i, :] = prob

    for _ in range(step):
        log = np.zeros((customer_num, customer_num), dtype=int)
        for i in range(customer_num):
            index = np.random.choice(customer_num, p=D[i])
            log[i, index], log[index, i] = 1, 1

        def rm(now, cust):
            cust.remove(now)
            conn = np.where(log[now, :] == 1)[0]
            for c in conn:
                if not c in cust:
                    continue
                else:
                    rm(c, cust)

        cust = [i for i in range(customer_num)]
        while cust:
            c = cust[0]
            rm(c, cust)
            table_num += 1

    return table_num/step


# multiprocess
def mp_ddcrp(customer: np.array, alpha=5, a=3, step=100, worker=10):
    pool = mp.Pool(worker)
    multi_res = [pool.apply_async(ddcrp, (customer, alpha, a, step)) for _ in range(worker)]
    pool.close()
    pool.join()
    table_num = [res.get() for res in multi_res]
    table_num = np.array(table_num).mean()
    return table_num


if __name__ == '__main__':
    table_num = mp_ddcrp(X)
    print(table_num)
