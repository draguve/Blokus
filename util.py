import random
import string
from functools import wraps
import time
import os


def get_timestamp():
    return time.time()


def check_for_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def shuffle_together(list1, list2):
    zipped = list(zip(list1, list2))
    random.shuffle(zipped)
    list1, list2 = zip(*zipped)
    return list1, list2


def timeit(func, print_args=False):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        if print_args:
            print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        else:
            print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        return result

    return timeit_wrapper


def random_id(n=10):
    return ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=n))
