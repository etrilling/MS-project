import numpy as np
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt
from derivative import dxdt
from tqdm import tqdm


# set random seed for reproducibility
np.random.seed(0)

# set global solve_ivp keyword arguments
solve_ivp_kwargs = {}
solve_ivp_kwargs['method'] = 'LSODA'
solve_ivp_kwargs['rtol'] = 1e-6
solve_ivp_kwargs['atol'] = 1e-6
# solve_ivp_kwargs['rtol'] = 1e-12
# solve_ivp_kwargs['atol'] = 1e-12


# generate Lotka-Volterra right-hand side functions
def generate_lotka_volterra_rhs(a, b, c, d):
    def lotka_volterra_rhs(t, x):
        # we expect x to be a 2D array with shape (rows, cols) == (2, n_samples)
        assert x.shape[0] == 2
        return np.array([a*x[0] - b*x[0]*x[1], -c*x[1] + d*x[0]*x[1]])
    return lotka_volterra_rhs


# generate training data for an arbitrary right-hand side function
def generate_training_data(rhs, x0, t_eval):
    t_span = (t_eval[0], t_eval[-1])
    x = solve_ivp(rhs, t_span, x0, t_eval=t_eval, **solve_ivp_kwargs).y
    # NOTE: x.shape == (n_state_vars, n_samples) == (rows, cols)
    return x


# generate noisy training data
def add_noise_to_data(x, noise_level):
    # set random seed for reproducibility
    np.random.seed(0)

    # calculate the root-mean-square (RMS) value of each state variable
    rms = np.sqrt(np.mean(x**2, axis=1)).reshape((-1, 1))
    
    # calculate the standard deviation of the noise to add
    std = rms * noise_level
    
    # generate appropriate noise by scaling samples from a standard normal
    # NOTE: because "std" is a vector it gets broadcast to the shape of the standard normal noise
    noise = (std * np.random.standard_normal(x.shape))
    
    return x + noise


# generate model predictions
def generate_model_prediction(model, x0, t_eval):
    def model_rhs(t, x):
        return model.predict_single(x)
    
    t_span = (t_eval[0], t_eval[-1])

    # attempt to solve the IVP problem
    # if we fail to solve the IVP problem, return None
    # NOTE: this can happen if the model predicts a trajectory that blows up or simply is very stiff
    try:
        ivp_result = solve_ivp(model_rhs, t_span, x0, t_eval=t_eval, **solve_ivp_kwargs)
    except ValueError:
        return None
    
    if ivp_result.success is False:
        return None
    
    return ivp_result.y


# calculate the relative error between the true and predicted trajectories using the frobenius norm
def relative_trajectory_error(x_train, x_pred):
    assert x_train.shape == x_pred.shape
    return np.sum((x_train - x_pred)**2) / np.sum(x_train**2)


# calculate the relative error between the true and predicted coefficients using the frobenius norm
def relative_coefficient_error(true_coeffs, pred_coeffs):
    assert true_coeffs.shape == pred_coeffs.shape
    return np.sum((true_coeffs - pred_coeffs)**2) / np.sum(true_coeffs**2)
