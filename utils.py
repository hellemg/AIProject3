import time
import numpy as np


def test_time(method, args, num_runs=100):
    runs = np.ones(num_runs)
    print('...doing {} runs'.format(num_runs))
    for i in range(num_runs):
        start_time = time.time()
        method(*args)
        end_time = time.time()
        runs[i] = end_time-start_time
    time_used = np.mean(runs)
    print('...method takes {} seconds'.format(time_used))
    return time_used
