import numpy as np
from scipy.integrate import solve_ivp
import time


# set global solve_ivp keyword arguments
# NOTES
# - When I run RK45 with a timeout, it gives very close to the same results as LSODA but a little slower due to the
#   timeouts. Radau is very slow but produces very close to the same results as LSODA.
#   When Radau fails (on the same problems LSODA does), the status message is
#   "Required step size is less than spacing between numbers" which is more informative than what LSODA gives.
# - Using the default tolerances (for LSODA at least) produces pretty bad results. However, there doesn't seem to
#   be a huge difference between 1e-6 and 1e-12.
solve_ivp_kwargs = {}
solve_ivp_kwargs['method'] = 'LSODA'
# solve_ivp_kwargs['rtol'] = 1e-12
# solve_ivp_kwargs['atol'] = 1e-12
solve_ivp_kwargs['rtol'] = 1e-6
solve_ivp_kwargs['atol'] = 1e-6


# custom exception for when the time limit is exceeded
class TimeoutError(Exception):
    pass


# wrapper around solve_ivp that adds a time limit to the integrationx
def solve_ivp_with_timeout(fun, t_span, y0, time_limit=10, **kwargs):
    # Record the start time
    start_time = time.time()

    # Custom event function to check if a time limit is exceeded
    def time_event(t, y):
        elapsed_time = time.time() - start_time

        if elapsed_time > time_limit:
            raise TimeoutError(f"Time limit of {time_limit} seconds exceeded. Elapsed time: {elapsed_time} seconds.")
        
        # NOTE: event functions must return a value that crosses zero when the event occurs
        return time_limit - elapsed_time
    
    # NOTE: these could be used, but they're not needed for my hacky solution of raising an exception
    # time_event.terminal = True  # Stop integration if the event triggers
    # time_event.direction = -1   # Detect if the output of time_event goes from positive to negative (crossing zero)
    
    ode_result = solve_ivp(fun, t_span, y0, **kwargs, events=[time_event])

    return ode_result


# generate training data for an arbitrary right-hand side function
def generate_training_data(rhs, x0, t_eval):
    t_span = (t_eval[0], t_eval[-1])
    x = solve_ivp_with_timeout(rhs, t_span, x0, t_eval=t_eval, **solve_ivp_kwargs).y
    return x


# generate model predictions
def generate_model_prediction(model, x0, t_eval):
    def model_rhs(t, x):
        return model.predict_single(x)
    
    t_span = (t_eval[0], t_eval[-1])

    # attempt to solve the IVP problem
    # if we fail to solve the IVP problem, return None
    # this can happen if the model predicts a trajectory that blows up or simply is very stiff
    try:
        ode_result = solve_ivp_with_timeout(model_rhs, t_span, x0, t_eval=t_eval, **solve_ivp_kwargs)
    except TimeoutError:
        print('timeout occurred in solve_ivp_with_timeout')
        return None
    except ValueError:
        # NOTE: it's possible this only happens for non-LSODA methods
        raise ValueError('hey Elliot, come check this out!')
        return None
    
    if ode_result.success is False:
        return None
    
    return ode_result.y


# generate noisy training data
def add_noise_to_data(x, noise_level, seed=0):
    # set random seed for reproducibility
    np.random.seed(seed)

    # calculate the root-mean-square (RMS) value of each state variable
    rms = np.sqrt(np.mean(x**2, axis=1)).reshape((-1, 1))
    
    # calculate the standard deviation of the noise to add
    std = rms * noise_level
    
    # generate appropriate noise by scaling samples from a standard normal
    # NOTE: because "std" is a vector it gets broadcast to the shape of the standard normal noise
    noise = std * np.random.standard_normal(x.shape)
    
    return x + noise


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


# define a function to interpolate the data using linear interpolation in multiple dimensions
def lineaer_interp(eval_pts, t_eval, x_train):
    # ensure the evaluation points are within the range of the training data
    assert min(eval_pts) >= min(t_eval) and max(eval_pts) <= max(t_eval)

    x_interp = np.zeros((x_train.shape[0], len(eval_pts)))

    # for each state variable, interpolate the data
    for i in range(x_train.shape[0]):
        x_interp[i] = np.interp(eval_pts, t_eval, x_train[i])
    
    return x_interp
