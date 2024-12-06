import numpy as np


# NOTE: the following code was generated with "o1-preview"
# TODO: look over this code and make sure you understand it!
def dxdt_finite_difference_1d(x, t):
    """
    Compute the derivative of x with respect to t using:
    - Second-order accurate central differences in the interior.
    - First-order accurate one-sided differences at the boundaries.
    
    This function matches the behavior of np.gradient(x, t, edge_order=1).
    
    Parameters:
    x (array_like): Array of function values.
    t (array_like): Array of times or positions corresponding to x.
    
    Returns:
    ndarray: Array of derivative values dx/dt.
    """
    assert x.ndim == 1
    assert t.ndim == 1

    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    
    if x.shape != t.shape:
        raise ValueError("x and t must have the same shape.")
    
    dxdt = np.zeros_like(x)
    
    # First point: forward difference
    dxdt[0] = (x[1] - x[0]) / (t[1] - t[0])
    
    # Interior points
    dt1 = t[1:-1] - t[:-2]    # t_i - t_{i-1}
    dt2 = t[2:] - t[1:-1]     # t_{i+1} - t_i
    dx1 = x[1:-1] - x[:-2]    # x_i - x_{i-1}
    dx2 = x[2:] - x[1:-1]     # x_{i+1} - x_i
    
    denom = dt1 + dt2
    a = dt2 / denom
    b = dt1 / denom
    
    dxdt[1:-1] = a * (dx1 / dt1) + b * (dx2 / dt2)
    
    # Last point: backward difference
    dxdt[-1] = (x[-1] - x[-2]) / (t[-1] - t[-2])
    
    return dxdt


def dxdt_finite_difference(x, t):
    # NOTE: this gives exactly the same result as "dxdt(x, t, kind="finite_difference", k=1)"
    # NOTE: this also gives the same result as "np.gradient(x, t, edge_order=1, axis=1)"
    #       see the page "https://numpy.org/doc/stable/reference/generated/numpy.gradient.html" for more information

    # we will assume that x.shape == (n_state_vars, n_samples)
    
    # make sure x and t have the same number of time points
    assert x.shape[1] == len(t)

    dxdt = np.zeros_like(x)

    for i in range(x.shape[0]):
        dxdt[i] = dxdt_finite_difference_1d(x[i], t)
    
    return dxdt



# def dxdt_es_finite_difference_1d(x, t):
#     assert x.ndim == 1
#     assert t.ndim == 1
#     assert x.shape == t.shape

#     dxdt = np.zeros_like(x)
#     dxdt[0] = (x[1] - x[0]) / (t[1] - t[0])
#     dxdt[1:-1] = (x[2:] - x[:-2]) / (t[2:] - t[:-2])
#     dxdt[-1] = (x[-1] - x[-2]) / (t[-1] - t[-2])
#     return dxdt


# def dxdt_es_finite_difference(x, t):
#     assert x.shape[1] == len(t)

#     dxdt = np.zeros_like(x)

#     for i in range(x.shape[0]):
#         dxdt[i] = dxdt_es_finite_difference_1d(x[i], t)
    
#     return dxdt



def dxdt_poly_fit_1d(x, t, poly_deg, window_diameter):
    # NOTE: this gives exactly the same result as
    #       "dxdt(x, t, kind="savitzky_golay", left=window_diameter, right=window_diameter, order=poly_deg, iwindow=True)"
    # NOTE: window_diameter is the number of points on each side of the point to fit the polynomial to.
    #       If there are not enough points on one side, it will use all the points on that side.

    assert x.ndim == 1
    assert t.ndim == 1

    dxdt = np.zeros_like(x)

    for i in range(len(x)):
        if i < window_diameter:
            start = 0
            end = i + window_diameter + 1 # +1 because the end index is exclusive
        elif i + window_diameter >= len(x):
            start = i - window_diameter
            end = len(x)
        else:
            start = i - window_diameter
            end = i + window_diameter + 1 # +1 because the end index is exclusive

        # fit a polynomial to the data
        p = np.polyfit(t[start:end], x[start:end], poly_deg)

        # calculate the derivative of the polynomial at the current time point
        dxdt[i] = np.polyval(np.polyder(p), t[i])
    
    return dxdt


def dxdt_poly_fit(x, t, poly_deg, window_diameter):
    # we will assume that x.shape == (n_state_vars, n_samples)
    
    # make sure x and t have the same number of time points
    assert x.shape[1] == len(t)

    dxdt = np.zeros_like(x)

    for i in range(x.shape[0]):
        dxdt[i] = dxdt_poly_fit_1d(x[i], t, poly_deg, window_diameter)
    
    return dxdt
