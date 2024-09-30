import numpy as np

from src.SINDyModel import make_poly_library, make_fourier_library


# generate the true coefficients matrix for some right-hand side function
# NOTE: this function should only be used within this file as a helper
def generate_true_coefficients(coefficient_info, poly_order=5, n_frequencies=0):
    # NOTE: coefficient_info contains info for the RHS of each state variable
    #       it is a list of dictionaries where each dictionary has the format
    #       key: function_name, value: coefficient_value
    
    n_state_vars = len(coefficient_info)

    # generate the function library and names
    function_library = []
    function_library_names = []
    library_functions, library_names = make_poly_library(n_state_vars, poly_order)
    function_library += library_functions
    function_library_names += library_names
    library_functions, library_names = make_fourier_library(n_state_vars, n_frequencies)
    function_library += library_functions
    function_library_names += library_names

    # generate the true coefficients matrix
    true_coeffs = np.zeros((len(function_library), n_state_vars))

    # fill in the true coefficients for each state variable using the info in coefficient_info
    for state_var in range(n_state_vars):
        for i, function_name in enumerate(function_library_names):
            true_coeffs[i, state_var] = coefficient_info[state_var].get(function_name, 0)
    
    return true_coeffs


# generate Lotka-Volterra right-hand side functions
def generate_lotka_volterra_rhs(a, b, c, d):
    def lotka_volterra_rhs(t, x):
        # we expect x to be a 2D array with shape (rows, cols) == (2, n_samples)
        assert x.shape[0] == 2
        return np.array([a*x[0] - b*x[0]*x[1], -c*x[1] + d*x[0]*x[1]])

    return lotka_volterra_rhs

# generate true coefficients for the Lotka-Volterra system
def generate_lotka_volterra_true_coefficients(a, b, c, d, poly_order=5, n_frequencies=0):
    coefficient_info = [{'x0': a, 'x0 x1': -b}, # RHS for first state variable
                        {'x1': -c, 'x0 x1': d}] # RHS for second state variable

    true_coefficients = generate_true_coefficients(coefficient_info, poly_order, n_frequencies)

    return true_coefficients


def generate_van_der_pol_rhs(mu):
    def van_der_pol_rhs(t, x):
        # we expect x to be a 2D array with shape (rows, cols) == (2, n_samples)
        assert x.shape[0] == 2
        return np.array([x[1], -x[0] + mu*x[1] - mu*x[1]*x[0]**2])

    return van_der_pol_rhs

def generate_van_der_pol_true_coefficients(mu, poly_order=5, n_frequencies=0):
    coefficient_info = [{'x1': 1}, # RHS for first state variable
                        {'x0': -1, 'x1': mu, 'x0^2 x1': -mu}] # RHS for second state variable

    true_coefficients = generate_true_coefficients(coefficient_info, poly_order, n_frequencies)

    return true_coefficients
