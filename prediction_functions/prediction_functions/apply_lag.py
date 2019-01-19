import numpy as np
def apply_lag(x, y, lag):
    """
    Roll y by lag (-ve indicates y leads) and return overlapping points

    :param x: Input x values (numpy 1D array)
    :param y: Input y values (numpy 1D array)
    :param lag: Integer number of time steps -ve indicates y leads x
    :return: x_common,y_common,correlation_coefficient
    """
    # shift y by the lag and select only overlapping points
    nx = len(x)
    if lag < 0:
        idx_common = np.arange(-lag, nx, 1)
    else:
        idx_common = np.arange(0, nx - lag)
    y_common = np.roll(y, -lag)[idx_common]
    x_common = x[idx_common]
    correlation_coefficient = np.corrcoef(x_common, y_common)[0, 1]
    return (x_common,y_common,correlation_coefficient)

