from numba import njit


@njit
def clip_nb(x, x_min, x_max):
    assert x_min <= x_max, (x_min, x_max)
    if x < x_min:
        return x_min
    elif x > x_max:
        return x_max
    return x
