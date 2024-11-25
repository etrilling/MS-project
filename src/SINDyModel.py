import numpy as np
import itertools


def make_poly_library(n_state_vars, poly_order):
    def make_poly_function(power_tuple):
        # here power is a tuple of the powers of the state variables
        # e.g. (1, 2) means x0^1 * x1^2
        return lambda x: np.prod(np.power(x, power_tuple), axis=1)
    
    # get all possible power tuples up to poly_order
    # this should be sorted so that all n-th order terms come before all (n+1)-th order terms
    # and within each order, the terms are sorted lexicographically by the power of the state variables
    power_tuples = []
    power_tuples.append((0,) * n_state_vars) # add the constant term
    for n in range(1, poly_order + 1):
        power_tuples += sorted([p for p in itertools.product(range(n + 1), repeat=n_state_vars) if sum(p) == n],
                               reverse=True)
    
    # create the library functions and their names
    library_functions = []
    library_names = []
    for power_tuple in power_tuples:
        library_functions.append(make_poly_function(power_tuple))

        # create a name for the library function based on the powers of the state variables
        function_name = ""
        for i in range(n_state_vars):
            if power_tuple[i] == 0:
                continue
            if power_tuple[i] == 1:
                function_name += f"x{i} "
            else:
                function_name += f"x{i}^{power_tuple[i]} "
        function_name = function_name.strip() # remove the trailing space

        # add the function name to the library names
        library_names.append(function_name)
    
    library_names[0] = "1" # replace the first name to make it more clear that it is the constant term
    
    return library_functions, library_names


def make_fourier_library(n_state_vars, n_frequencies, include_sin=True, include_cos=True):
    def make_sin_function(freq, i):
        return lambda x: np.sin(freq*x[:, i])

    def make_cos_function(freq, i):
        return lambda x: np.cos(freq*x[:, i])
    
    library_functions = []
    library_names = []
    for freq in range(1, n_frequencies + 1): # NOTE: we do not include zero frequency terms
        for i in range(n_state_vars):
            # add the sin and cos functions to the library
            if include_sin:
                library_functions.append(make_sin_function(freq, i))
                library_names.append(f"sin({freq}*x{i})")
            if include_cos:
                library_functions.append(make_cos_function(freq, i))
                library_names.append(f"cos({freq}*x{i})")

    return library_functions, library_names


class SINDyModel:
    def __init__(self, n_state_vars, threshold, poly_order=5, n_frequencies=0):
        self.n_state_vars = n_state_vars
        self.threshold = threshold
        
        # add functions to the function library
        self.function_library = []
        self.function_library_names = []
        library_functions, library_names = make_poly_library(self.n_state_vars, poly_order)
        self.function_library += library_functions
        self.function_library_names += library_names
        library_functions, library_names = make_fourier_library(self.n_state_vars, n_frequencies)
        self.function_library += library_functions
        self.function_library_names += library_names

    
    def fit(self, x, x_dot, max_itterations=20):
        # do some basic checks on the input data shapes
        assert x.shape[0] == self.n_state_vars
        assert x.shape == x_dot.shape

        # transpose the data to the form (rows, cols) == (n_samples, n_functions)
        # NOTE: it is just easier to work with the data in this form here
        x = x.T
        x_dot = x_dot.T
        
        Theta = np.column_stack([f(x) for f in self.function_library])

        Xi = np.linalg.lstsq(Theta, x_dot, rcond=None)[0]

        # repeat STLSQ multiple times
        for _ in range(max_itterations):
            # get indices of small coefficients
            small_inds = np.abs(Xi) < self.threshold

            # check if there are any small (but non-zero) coefficients
            # if there are any, we should keep going. if there are none, we can stop.
            keep_going = np.any(small_inds & (Xi != 0))
            if not keep_going:
                break

            # set the small coefficients to zero
            Xi[small_inds] = 0

            # compute least squares on each state variable on the remaining terms
            for i in range(x.shape[1]):
                # get the indices of the non-zero coefficients for this state variable
                big_inds = ~small_inds[:, i]
                # compute least squares
                Xi[big_inds, i] = np.linalg.lstsq(Theta[:, big_inds], x_dot[:, i], rcond=None)[0]

                # NOTE: the following code is an alternative to the above line
                # from sklearn.linear_model import ridge_regression
                # Xi[big_inds, i] = ridge_regression(Theta[:, big_inds], x_dot[:, i], alpha=0.05)
        else:
            raise RuntimeError("STLSQ did not converge in the maximum number of itterations")

        # save the Xi matrix
        self.Xi = Xi


    def predict(self, x):
        # NOTE: "x" will take the form (n_state_vars, n_samples)
        #       so, x.T will take the form (n_samples, n_state_vars)

        # predict the time derivatives of the state variables
        x = x.T
        Theta = np.column_stack([f(x) for f in self.function_library])
        x_dot = Theta @ self.Xi
        return x_dot.T
    
    
    def predict_single(self, x):
        # NOTE: "x" will take the form (n_state_vars,)
        return self.predict(x.reshape(-1, 1)).flatten()
    

    def print(self, decimals=5):
        # TODO (very minor): work on formatting the output a little better

        # print the learned model
        for i in range(self.Xi.shape[1]):
            print(f"x{i}' = ", end="") # NOTE: we use "i + 1" because the state variables are 1-indexed

            for j in range(self.Xi.shape[0]):
                if self.Xi[j, i] != 0:
                    print("+ ", end="")
                    print(f"{self.Xi[j, i]:.{decimals}f} {self.function_library_names[j]}", end=" ")
            print()
