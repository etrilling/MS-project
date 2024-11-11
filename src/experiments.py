from src.helpers import *
from src.SINDyModel import SINDyModel
from src.dynamical_systems import *

from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns


# define the keys that are fixed in the experiment configuration
# NOTE: "true_coeffs" is not used in the experiment but having it in the configuration makes generating results easier
fixed_keys = ['rhs', 'derivative_approximation', 't_eval_test', 'np_seed', 'true_coeffs']

# define the keys that can change in the experiment configuration
dynamic_keys = ['x0', 'noise_level',
                't_eval_train', 't_span_train', 'dt_train', # either t_eval_train or (t_span_train and dt_train)
                'train_interpolation_dt', # can either be None or a float
                'threshold', 'poly_order', 'n_frequencies']


def run_experiment(experiment_config, generate_prediction=True):
    # NOTE: the order of the keys in "experiment_config" is important.
    #       the first list present will be the rows and the second list will be the columns

    # check that "experiment_config" has exactly the keys we expect
    assert set(experiment_config.keys()) == set(fixed_keys + dynamic_keys)

    # extract the parts of the experiment configuration that don't change
    rhs = experiment_config['rhs']
    derivative_approximation = experiment_config['derivative_approximation']
    t_eval_test = experiment_config['t_eval_test']
    np_seed = experiment_config['np_seed']
    
    # figure out which two parameters are lists
    list1 = None
    list2 = None
    for key in experiment_config.keys():
        if type(experiment_config[key]) is not list:
            continue

        if list1 is None:
            list1 = experiment_config[key]
            print(f'rows are {key}. there are {len(list1)} rows.')
        elif list2 is None:
            list2 = experiment_config[key]
            print(f'columns are {key}. there are {len(list2)} columns.')
        else:
            raise ValueError("Only two lists are allowed")
    
    # make sure we found two lists
    if list1 is None or list2 is None:
        raise ValueError("You need to specify two lists in the experiment configuration")

    # make a results matrix
    # the first list (list1) will decide be the number of rows and
    # the second list (list2) will decide the number of columns
    results = [[None for _ in range(len(list2))] for _ in range(len(list1))]

    # iterate over the lists
    for i in range(len(list1)): # iterate over the rows
        print(f'step {i + 1} of {len(list1)}')

        for j in tqdm(range(len(list2))): # iterate over the columns
            result = {}

            # iterate over the experiment configuration and set values of result for this iteration
            for key in experiment_config.keys():
                # skip the keys that we are not interested in (this is needed to keep excess data out of the results)
                if key in fixed_keys:
                    continue

                # if the key is a list, we need to get the value for this iteration
                if type(experiment_config[key]) is list:
                    # figure out which index to use for this list
                    # we will index list1 by i and list2 by j
                    idx = i if experiment_config[key] is list1 else j
                    result[key] = experiment_config[key][idx]
                else:
                    result[key] = experiment_config[key]
            
            # set local variables
            x0 = result['x0']
            noise_level = result['noise_level']
            t_eval_train = result['t_eval_train']
            dt_train = result['dt_train']
            t_span_train = result['t_span_train']
            threshold = result['threshold']
            poly_order = result['poly_order']
            n_frequencies = result['n_frequencies']
            train_interpolation_dt = result['train_interpolation_dt']

            # calculate t_eval_train if it is None
            if t_eval_train is None:
                if dt_train is None or t_span_train is None:
                    raise ValueError("You need to specify either t_eval_train or (dt_train and t_span_train)")
                t_eval_train = np.arange(t_span_train[0], t_span_train[1] + dt_train, dt_train)
                result['t_eval_train'] = t_eval_train
            else:
                if dt_train is not None or t_span_train is not None:
                    raise ValueError("You can't specify both t_eval_train and (dt_train or t_span_train)")
            
            # generate training data
            x_train = generate_training_data(rhs, x0, t_eval_train)
            result["x_train"] = x_train
            
            # generate noisy data
            x_train_noisy = add_noise_to_data(x_train, noise_level, np_seed)
            result["x_train_noisy"] = x_train_noisy

            # interpolate the noisy data if needed
            # NOTE: we do not save the interpolated data in the results object, it is just used here
            if train_interpolation_dt is not None:
                eval_pts = np.arange(t_eval_train[0], t_eval_train[-1], train_interpolation_dt)
                x_train_noisy = linear_interp(eval_pts, t_eval_train, x_train_noisy)
                t_eval_train = eval_pts
            
            # generate a derivative approximation of the noisy data
            x_dot_approx = derivative_approximation(x_train_noisy, t_eval_train)
            result["x_dot_approx"] = x_dot_approx

            # generate the true derivative of the data
            # this is only used to generate results and is not used in the SINDy model
            x_dot_true = rhs(0, x_train)
            result["x_dot_true"] = x_dot_true
            
            # create and fit a SINDy model
            model = SINDyModel(n_state_vars=len(x0), threshold=threshold,
                               poly_order=poly_order, n_frequencies=n_frequencies)
            model.fit(x_train_noisy, x_dot_approx)
            result["model"] = model

            if generate_prediction:
                # generate the model prediction on the test data
                x_pred = generate_model_prediction(model, x0, t_eval_test)
                result["x_pred"] = x_pred

                # generate the true test data
                x_test = generate_training_data(rhs, x0, t_eval_test)
                result["x_test"] = x_test
            
            # store the data entry
            results[i][j] = result

    return results



def plot_heatmaps(experiment_config, results):
    # figure out which parameters are lists
    list1 = None
    list1_name = None
    list2 = None
    list2_name = None
    for key in experiment_config.keys():
        if type(experiment_config[key]) is not list:
            continue

        if list1 is None:
            list1 = experiment_config[key]
            list1_name = key
            print(f'rows are {key}. there are {len(list1)} rows.')
        elif list2 is None:
            list2 = experiment_config[key]
            list2_name = key
            print(f'columns are {key}. there are {len(list2)} columns.')
        else:
            raise ValueError("Only two lists are allowed")
    
    # flip the rows of the results so the smallest value is at the bottom of the heatmap
    results = results[::-1]
    
    # generate the labels for the heatmap using the list values
    yticklabels = [f"{val:.4f}\n{i}" if type(val) is float else f"{val}\n{i}"
                   for i, val in list(enumerate(list1))[::-1]] # row labels
    xticklabels = [f"{val:.4f}\n{i}" if type(val) is float else f"{val}\n{i}"
                   for i, val in enumerate(list2)] # column labels

    # extract the true coefficients from the experiment configuration
    true_coeffs = experiment_config['true_coeffs']

    # make a matrix of the relative coefficient errors
    rce_mtx = np.zeros((len(list1), len(list2)))
    for i in range(len(list1)):
        for j in range(len(list2)):
            model = results[i][j]["model"]
            rce = relative_coefficient_error(true_coeffs, model.Xi)
            rce_mtx[i, j] = rce

    # make a matrix of the relative trajectory errors
    rte_mtx = np.zeros((len(list1), len(list2)))
    for i in range(len(list1)):
        for j in range(len(list2)):
            x_test = results[i][j].get("x_test", None) # use "get" so this works when we don't generate predictions
            x_pred = results[i][j].get("x_pred", None)
            
            # if the model prediction is "None", set the rte to np.nan
            if x_pred is None:
                rte = np.nan
            else:
                rte = relative_trajectory_error(x_test, x_pred)
            
            rte_mtx[i, j] = rte

    # draw the plots
    sns.set_theme(font_scale=0.75)

    fig = plt.figure(figsize=(8, 6))
    ax = sns.heatmap(rce_mtx, annot=True, xticklabels=xticklabels, yticklabels=yticklabels, vmin=0, vmax=1)
    ax.set(xlabel=list2_name, ylabel=list1_name, title="Relative Coefficient Error")

    fig = plt.figure(figsize=(8, 6))
    ax = sns.heatmap(rte_mtx, annot=True, xticklabels=xticklabels, yticklabels=yticklabels, vmin=0, vmax=1)
    ax.set(xlabel=list2_name, ylabel=list1_name, title="Relative Trajectory Error")

    plt.show()



def display_single_result(experiment_config, results, col, row, keys_to_display=['x_test', 'x_pred', 'x_train_noisy']):
    result = results[row][col]

    # print the model for this result
    model = result['model']
    model.print()

    # get the number of state variables
    n_state_vars = len(experiment_config['x0'])

    fig, axs = plt.subplots(1, n_state_vars, figsize=(16, 6), sharex='col', sharey='row')

    for key in keys_to_display:
        # figure out which time points to use
        if key in ('x_train', 'x_train_noisy', 'x_dot_approx', 'x_dot_true'):
            t_eval = result['t_eval_train']
            marker = '.-'
        if key in ('x_test', 'x_pred'):
            # NOTE: we don't store t_eval_test in the results because it is the same for all results
            t_eval = experiment_config['t_eval_test']
            marker = '-'
        
        # extract the data
        x = result[key]

        # plot each state variable
        for sv in range(n_state_vars):
            axs[sv].plot(t_eval, x[sv], marker, label=key)

            # set labels
            axs[sv].set_xlabel("t")
            axs[sv].set_ylabel(f"$x_{sv}$")
            axs[sv].legend()



def run_test_suite(experiment_config, generate_prediction=True):
    # TODO: you'd need to change the library functions for the pendulum

    list_of_rhs = [
        generate_lotka_volterra_rhs(*default_params['lotka_volterra']),
        generate_van_der_pol_oscillator_rhs(*default_params['van_der_pol_oscillator']),
        generate_duffing_oscillator_rhs(*default_params['duffing_oscillator']),
        generate_nonlinear_pendulum_rhs(*default_params['nonlinear_pendulum']),
        generate_lorenz_rhs(*default_params['lorenz'])
    ]
    
    list_of_true_coeffs = [
        generate_lotka_volterra_true_coefficients(*default_params['lotka_volterra']),
        generate_van_der_pol_oscillator_true_coefficients(*default_params['van_der_pol_oscillator']),
        generate_duffing_oscillator_true_coefficients(*default_params['duffing_oscillator']),
        generate_nonlinear_pendulum_true_coefficients(*default_params['nonlinear_pendulum']),
        generate_lorenz_true_coefficients(*default_params['lorenz'])
    ]
    
    list_of_x0 = [
        default_x0['lotka_volterra'],
        default_x0['van_der_pol_oscillator'],
        default_x0['duffing_oscillator'],
        default_x0['nonlinear_pendulum'],
        default_x0['lorenz']
    ]
    
    list_of_system_names = [
        'Lotka-Volterra',
        'Van der Pol Oscillator',
        'Duffing Oscillator',
        'Nonlinear Pendulum',
        'Lorenz System'
    ]

    list_of_results = []
    for rhs, true_coeffs, x0, system_name in zip(list_of_rhs, list_of_true_coeffs, list_of_x0, list_of_system_names):
        experiment_config['rhs'] = rhs
        experiment_config['true_coeffs'] = true_coeffs
        experiment_config['x0'] = x0

        results = run_experiment(experiment_config, generate_prediction)
        list_of_results.append(results)

        print('*'*30 + system_name + '*'*30)
        plot_heatmaps(experiment_config, results)
    

    return list_of_results
