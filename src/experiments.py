from src.helpers import *
from src.SINDyModel import SINDyModel
from src.ESINDyModel import ESINDyModel
from src.dynamical_systems import *
from src.derivatives import *

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


def extract_list_items(experiment_config):
    list_kv_pairs = list(filter(lambda kv_pair: type(kv_pair[1]) is list, experiment_config.items()))
    
    if len(list_kv_pairs) != 2:
        raise ValueError("You need to specify two lists in the experiment configuration")
    
    list1_name = list_kv_pairs[0][0]
    list1 = list_kv_pairs[0][1]
    list2_name = list_kv_pairs[1][0]
    list2 = list_kv_pairs[1][1]

    return list1_name, list1, list2_name, list2


def run_experiment(experiment_config, use_ESINDy=False, generate_prediction=True):
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
    list1_name, list1, list2_name, list2 = extract_list_items(experiment_config)
    print(f'rows are {list1_name}. there are {len(list1)} rows.')
    print(f'columns are {list2_name}. there are {len(list2)} columns.')

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
            if use_ESINDy:
                model = ESINDyModel(n_state_vars=len(x0), threshold=threshold,
                                    poly_order=poly_order, n_frequencies=n_frequencies)
            else:
                model = SINDyModel(n_state_vars=len(x0), threshold=threshold,
                                   poly_order=poly_order, n_frequencies=n_frequencies)
            model.fit(x_train_noisy, x_dot_approx)
            result["model"] = model
            result["model_xi"] = model.Xi

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


# def run_experiment_multiple_times(experiment_config, n_repeats, generate_prediction=True):
#     list_of_results = []

#     for i in range(n_repeats):
#         print('*'*20 + f'repeat {i + 1} of {n_repeats}' + '*'*20)
#         results = run_experiment(experiment_config, generate_prediction)
#         list_of_results.append(results)
    
#     list_1_len = len(list_of_results[0])
#     list_2_len = len(list_of_results[0][0])
#     avg_results = [[{} for _ in range(list_2_len)] for _ in range(list_1_len)]

#     for i in range(n_repeats):
#         for j in range(list_1_len):
#             for k in range(list_2_len):
#                 if avg_results[j][k] is None:
#                     avg_results[j][k]['model_xi'] = list_of_results[i][j][k]['model_xi']
#                     avg_results[j][k]['t'] = list_of_results[i][j][k]['x_train']
#                 else:
#                     avg_results[j][k] += list_of_results[i][j][k]





def plot_heatmaps(experiment_config, results):
    list1_name, list1, list2_name, list2 = extract_list_items(experiment_config)
    print(f'rows are {list1_name}. there are {len(list1)} rows.')
    print(f'columns are {list2_name}. there are {len(list2)} columns.')
    
    # flip the rows of the results so the smallest value is at the bottom of the heatmap
    results = results[::-1]
    
    # generate the labels for the heatmap using the list values
    yticklabels = [f"{val:.4f}\n{i}" if type(val) is float else f"{val}\n{i}"
                   for i, val in list(enumerate(list1))[::-1]] # row labels
    xticklabels = [f"{val:.4f}\n{i}" if type(val) is float else f"{val}\n{i}"
                   for i, val in enumerate(list2)] # column labels

    # extract the true coefficients from the experiment configuration
    true_coeffs = experiment_config['true_coeffs']

    # make a matrix of the relative coefficient errors and keep track of the correct systems
    rce_mtx = np.zeros((len(list1), len(list2)))
    correct_system_mtx = np.zeros((len(list1), len(list2)))
    for i in range(len(list1)):
        for j in range(len(list2)):
            model_xi = results[i][j]["model_xi"]
            rce = relative_coefficient_error(true_coeffs, model_xi)
            rce_mtx[i, j] = rce

            # if the predicted model has the same non-zero coefficients as the true model, set this to 1
            correct_system_mtx[i, j] = np.array_equal(np.nonzero(model_xi), np.nonzero(true_coeffs))

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
    # overlay the incorrect systems in red
    for i in range(len(list1)):
        for j in range(len(list2)):
            if correct_system_mtx[i, j] == False:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='blue', lw=1))
    
    # only draw the RTE plot if there are predictions
    if not np.all(np.isnan(rte_mtx)):
        fig = plt.figure(figsize=(8, 6))
        ax = sns.heatmap(rte_mtx, annot=True, xticklabels=xticklabels, yticklabels=yticklabels, vmin=0, vmax=1)
        ax.set(xlabel=list2_name, ylabel=list1_name, title="Relative Trajectory Error")
        # overlay the incorrect systems in red
        for i in range(len(list1)):
            for j in range(len(list2)):
                if correct_system_mtx[i, j] == False:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='blue', lw=1))

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



def run_test_suite(test_suite_config, use_ESINDy, generate_prediction):
    list_of_system_info = [
        ('lotka_volterra',
         generate_lotka_volterra_rhs(*default_params['lotka_volterra']),
         generate_lotka_volterra_true_coefficients(*default_params['lotka_volterra']),
         default_x0['lotka_volterra']),

        ('van_der_pol_oscillator',
        generate_van_der_pol_oscillator_rhs(*default_params['van_der_pol_oscillator']),
        generate_van_der_pol_oscillator_true_coefficients(*default_params['van_der_pol_oscillator']),
        default_x0['van_der_pol_oscillator']),

        ('duffing_oscillator',
        generate_duffing_oscillator_rhs(*default_params['duffing_oscillator']),
        generate_duffing_oscillator_true_coefficients(*default_params['duffing_oscillator']),
        default_x0['duffing_oscillator']),

        ('nonlinear_pendulum',
        generate_nonlinear_pendulum_rhs(*default_params['nonlinear_pendulum']),
        generate_nonlinear_pendulum_true_coefficients(*default_params['nonlinear_pendulum']),
        default_x0['nonlinear_pendulum']),

        ('lorenz',
        generate_lorenz_rhs(*default_params['lorenz']),
        generate_lorenz_true_coefficients(*default_params['lorenz']),
        default_x0['lorenz'])   
    ]

    list_of_results = []
    list_of_experiment_configs = []

    for system_name, rhs, true_coeffs, x0 in list_of_system_info:
        experiment_config = {
            # define fixed parameters
            'rhs': rhs,
            'derivative_approximation': dxdt_finite_difference,
            'np_seed': 0,
            'true_coeffs': true_coeffs,
            # define parameters that can be lists
            'x0': x0,
            't_eval_train': None,
            'train_interpolation_dt': None,
            'threshold': 0.05
        }

        if system_name == 'nonlinear_pendulum':
            experiment_config['poly_order'] = 1
            experiment_config['n_frequencies'] = 5
        else:
            experiment_config['poly_order'] = 5
            experiment_config['n_frequencies'] = 0
        
        keys = ['t_eval_test', 'noise_level',  'dt_train', 't_span_train']
        for key in keys:
            experiment_config[key] = test_suite_config[system_name][key]
        
        list_of_experiment_configs.append(experiment_config)

        print('*'*30 + system_name + '*'*30)
        results = run_experiment(experiment_config, use_ESINDy, generate_prediction)
        list_of_results.append(results)

        plot_heatmaps(experiment_config, results)
    

    return list_of_results, list_of_experiment_configs
