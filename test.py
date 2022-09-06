import pandas as pd
import itertools
from multiprocessing import Queue, Pool, cpu_count, Process, current_process, freeze_support
import numpy as np
import timeit

def worker_init(check_tmp):
    global check
    check = check_tmp
    process_name = current_process().name
    print(f'Worker {process_name}: Started working')


def f(a):
    return a*check

def basic(a,b):
    c = a | b
    return c


if __name__=='__main__':
    # Windows flag
    # freeze_support()
    # c = 1

    # pool = Pool(10, initializer=worker_init, initargs=(c,))
    # b = pool.map(f, list(range(20)))
    # print(b)
    # pool.close()
    # pool.join()
    # c = 2
    # pool = Pool(10, initializer=worker_init, initargs=(c,))
    # b = pool.map(f, list(range(20)))
    # print(b)
    # pool.close()
    # pool.join()

    # a = {'a': 1, 'b': 2, 'c': 3}
    # c = {'a': 1, 'b': 2, 'c': 3}
    # b = [a, c]
    # b[0]['e'] = 4
    # print(b)

    # expr = '(df[columns[1]]&~df[columns[0]])|(~df[columns[1]]&df[columns[0]])'
    # bool_pairs = []

    # for cnf in expr.replace(" ", "").split('|'):
    #     bool_pair = []
    #     print(cnf)

    #     for part in cnf.split('&'):
    #         print(part)
    #         if part[0] == '~':
    #             bool_pair.append(False)

    #         else:
    #             bool_pair.append(True)

    #     bool_pairs.append(np.array(bool_pair))

    # bool_pairs = np.array(bool_pairs)
    # print(bool_pairs)
    a = pd.Series(True, index=list(range(2000)))
    b = pd.Series(False, index=list(range(2000)))
    timeit.timeit(basic(a,b))