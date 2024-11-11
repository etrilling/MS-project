import numpy as np

from src.SINDyModel import make_poly_library, make_fourier_library


default_x0 = {}
default_params = {}


# generate the true coefficients matrix for some right-hand side function
# NOTE: this function should only be used within this file as a helper
def generate_true_coefficients(coefficient_info, poly_order, n_frequencies):
    # NOTE: "coefficient_info" contains info for the RHS of each state variable derivative
    #       it is a list of dictionaries where each dictionary has the format "function_name: coefficient_value"
        
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
        for func_idx, function_name in enumerate(function_library_names):
            true_coeffs[func_idx, state_var] = coefficient_info[state_var].get(function_name, 0)
    
    return true_coeffs


default_x0['lotka_volterra'] = (10, 10) # x0 = (pray, predator)
default_params['lotka_volterra'] = (1.1, 0.4, 0.4, 0.1) # from wikipedia

def generate_lotka_volterra_rhs(alpha, beta, gamma, delta):
    def rhs(t, x):
        assert x.shape[0] == 2 # we expect x to be a 2D array with shape (rows, cols) == (2, n_samples)
        return np.array([alpha*x[0] - beta*x[0]*x[1], -gamma*x[1] + delta*x[0]*x[1]])
    return rhs

def generate_lotka_volterra_true_coefficients(alpha, beta, gamma, delta, poly_order=5, n_frequencies=0):
    coefficient_info = [{'x0': alpha, 'x0 x1': -beta}, # RHS for first state variable
                        {'x1': -gamma, 'x0 x1': delta}] # RHS for second state variable

    true_coefficients = generate_true_coefficients(coefficient_info, poly_order, n_frequencies)
    return true_coefficients



default_x0['van_der_pol_oscillator'] = (0, 1) # x0 = (x, x_dot)
default_params['van_der_pol_oscillator'] = (2,) # random default param

def generate_van_der_pol_oscillator_rhs(mu):
    def rhs(t, x):
        assert x.shape[0] == 2 # we expect x to be a 2D array with shape (rows, cols) == (2, n_samples)
        return np.array([x[1], -x[0] + mu*x[1] - mu*x[1]*x[0]**2])
    return rhs

def generate_van_der_pol_oscillator_true_coefficients(mu, poly_order=5, n_frequencies=0):
    coefficient_info = [{'x1': 1}, # RHS for first state variable
                        {'x0': -1, 'x1': mu, 'x0^2 x1': -mu}] # RHS for second state variable

    true_coefficients = generate_true_coefficients(coefficient_info, poly_order, n_frequencies)
    return true_coefficients



default_x0['duffing_oscillator'] = (0, 1) # x0 = (x, x_dot)
default_params['duffing_oscillator'] = (0.2, 0.2, 1) # random default params

def generate_duffing_oscillator_rhs(delta, alpha, beta):
    def rhs(t, x):
        assert x.shape[0] == 2 # we expect x to be a 2D array with shape (rows, cols) == (2, n_samples)
        return np.array([x[1], -delta*x[1] - alpha*x[0] - beta*x[0]**3])
    return rhs

def generate_duffing_oscillator_true_coefficients(delta, alpha, beta, poly_order=5, n_frequencies=0):
    coefficient_info = [{'x1': 1}, # RHS for first state variable
                        {'x1': -delta, 'x0': -alpha, 'x0^3': -beta}] # RHS for second state variable

    true_coefficients = generate_true_coefficients(coefficient_info, poly_order, n_frequencies)
    return true_coefficients



default_x0['nonlinear_pendulum'] = (np.pi/4, 0) # x0 = (theta, theta_dot)
default_params['nonlinear_pendulum'] = (1,) # sensible default omega

def generate_nonlinear_pendulum_rhs(omega):
    def rhs(t, x):
        assert x.shape[0] == 2 # we expect x to be a 2D array with shape (rows, cols) == (2, n_samples)
        return np.array([x[1], -omega**2 * np.sin(x[0])])
    return rhs

def generate_nonlinear_pendulum_true_coefficients(omega, poly_order=0, n_frequencies=5):
    coefficient_info = [{'x1': 1}, # RHS for first state variable
                        {'sin(x1)': -omega**2}] # RHS for second state variable

    true_coefficients = generate_true_coefficients(coefficient_info, poly_order, n_frequencies)
    return true_coefficients



default_x0['lorenz'] = (-8, 7, 27) # standard lorenz x0
default_params['lorenz'] = (10, 28, 8/3) # standard lorenz params

def generate_lorenz_rhs(sigma, rho, beta):
    def rhs(t, x):
        assert x.shape[0] == 3 # we expect x to be a 2D array with shape (rows, cols) == (3, n_samples)
        return np.array([sigma*(x[1] - x[0]), x[0]*(rho - x[2]) - x[1], x[0]*x[1] - beta*x[2]])
    return rhs

def generate_lorenz_true_coefficients(sigma, rho, beta, poly_order=5, n_frequencies=0):
    coefficient_info = [{'x1': sigma, 'x0': -sigma}, # RHS for first state variable
                        {'x0': rho, 'x1': -1, 'x0 x2': -1}, # RHS for second state variable
                        {'x2': -beta, 'x0 x1': 1}] # RHS for third state variable

    true_coefficients = generate_true_coefficients(coefficient_info, poly_order, n_frequencies)
    return true_coefficients
