import os
import pickle
import random


def check_and_remove(filename):
    os.remove(filename) if os.path.exists(filename) else None


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    
def dump_pickle(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def unroll_double_dict(d):
    ls = [list(x.values()) for x in d.values()]
    return [l for sub_ls in ls for l in sub_ls]

def unroll_triple_dict(d):
    ls = [list(x.values()) for x in d.values()]
    ls = [list(x.values()) for sub_ls in ls for x in sub_ls]
    return [l for sub_ls in ls for l in sub_ls]


def safe_max(ls):
        return max(ls) if len(ls) > 0 else 0
    

def argmax_dict(d: dict) -> str:
    v = list(d.values())
    k = list(d.keys())
    return k[v.index(max(v))]


def list_intersection(ls_1: list, ls_2: list) -> list:
    return list(set(ls_1) & set(ls_2))
    

def times_between(start_period, end_period, focus_ls):
    """
    Given a list of interesting intervals in the form (start_foc, end_foc), computes the sum of the durations of all 
    and only the intervals that are between the start and end of the period.
    Assumption: start_foc <= end_focus
    """
    return sum([min(end_foc, end_period) - max(start_foc, start_period) for (start_foc, end_foc) in focus_ls if end_foc > start_period and start_foc < end_period])


def dispositions(ls: list, k: int) -> list:
    """
    Returns all the dispositions (combinations in which order is important, without repetitions 
    of elements) of groups of 'k' elements taken from the given list.
    """
    res = [[el] for el in ls]

    while len(res[0]) < k:
        # add another letter to each previous disposition
        next_res = []
        for r in res:
            for el in ls:
                next_res.append(r + [el])
        res = next_res
        
    return res


def random_toss(p: float, precision: int = 1e3) -> bool:
    """
    Random boolean outcome, on average it return True 'p' percent of the times.
    """
    a = random.randint(0, precision)
    return a < p * precision


def bound(x, min_val = 0, max_val = 1):
    """
    Bounds the given value between a minimum and a maximum value.
    """
    return min(max(x, min_val), max_val)
