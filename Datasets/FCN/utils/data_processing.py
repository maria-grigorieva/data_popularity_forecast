import numpy as np


def get_tightened_active_part(sample) -> np.ndarray:
    non_zero = np.nonzero(sample)
    i_begin = non_zero[0][0]
    i_end = non_zero[0][-1] + 1
    i0 = max(0, i_begin)
    i1 = min(i_end, len(sample) - 1)
    tight_sample = sample[i0 : i1]
    return tight_sample
