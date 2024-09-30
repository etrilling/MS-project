import numpy as np
from scipy.integrate import solve_ivp


# set global solve_ivp keyword arguments
# NOTES
# - RK45 stalls when it's run on some SINDy models and Radau is very slow but produces the same results as LSODA
#   when Radau fails on the same problems LSODA does, the status message is
#   "Required step size is less than spacing between numbers." which is more informative than what LSODA gives
# - using the default tolerances (for LSODA at least) produces pretty bad results. but there doesn't seem to be a huge
#   difference between 1e-6 and 1e-12.
solve_ivp_kwargs = {}
solve_ivp_kwargs['method'] = 'LSODA'
# solve_ivp_kwargs['rtol'] = 1e-12
# solve_ivp_kwargs['atol'] = 1e-12
solve_ivp_kwargs['rtol'] = 1e-6
solve_ivp_kwargs['atol'] = 1e-6


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
        # NOTE: it's possible this only happens for non-LSODA methods
        raise ValueError('hey Elliot, come check this out!')
        return None
    
    if ivp_result.success is False:
        return None
    
    return ivp_result.y


# calculate the relative error between the true and predicted values using the frobenius norm
def relative_frobenius_error(x_true, x_pred):
    assert x_true.shape == x_pred.shape
    return np.linalg.norm(x_true - x_pred, ord='fro') / np.linalg.norm(x_true, ord='fro')

# calculate the relative error between the true and predicted trajectories using the frobenius norm
def relative_trajectory_error(x_train, x_pred):
    return relative_frobenius_error(x_train, x_pred)

# calculate the relative error between the true and predicted coefficients using the frobenius norm
def relative_coefficient_error(true_coeffs, pred_coeffs):
    return relative_frobenius_error(true_coeffs, pred_coeffs)
