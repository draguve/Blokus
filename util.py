import random
import string
from functools import wraps
import time
import os


def check_for_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result

    return timeit_wrapper


def random_id(n=10):
    return ''.join(random.choices(string.ascii_uppercase + string.ascii_uppercase + string.digits, k=n))
