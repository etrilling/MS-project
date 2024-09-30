import numpy as np


# generate the true coefficients matrix for some right-hand side function
# NOTE: this should only be used within this file as a helper function
def generate_true_coefficients(coefficient_info, model):
    # NOTE: coefficient_info contains info for the RHS of each state variable
    #       it is a list of dictionaries where each dictionary has the format
    #       key: function_name, value: coefficient_value
    # NOTE: model is an instance of SINDyModel

    n_state_vars = len(coefficient_info)

    # get the function library and function library names from the model
    function_library = model.function_library
    function_library_names = model.function_library_names

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
def generate_lotka_volterra_true_coefficients(a, b, c, d, model):
    coefficient_info = [{'x0': a, 'x0 x1': -b}, # RHS for first state variable
                        {'x1': -c, 'x0 x1': d}] # RHS for second state variable

    true_coefficients = generate_true_coefficients(coefficient_info, model)

    return true_coefficients
