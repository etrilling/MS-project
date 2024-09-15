import numpy as np
import itertools


def make_poly_library(n_state_vars, poly_order):
    def make_poly_function(powers):
        # here power is a tuple of the powers of the state variables
        # e.g. (1, 2) means x^1 * y^2
        return lambda x: np.prod(np.power(x, powers), axis=1)
    
    # get all possible power tuples up to poly_order
    powers = [power for power in itertools.product(range(poly_order + 1), repeat=n_state_vars) if sum(power) <= poly_order]
    
    # make the library functions
    library_functions = [make_poly_function(power) for power in powers]

    # make names for the library functions
    library_names = [" ".join([f"x{i}^{power[i]}" for i in range(n_state_vars) if power[i] != 0]) for power in powers]
    library_names[0] = "1" # replace the first name with a constant term
    
    return library_functions, library_names


def make_fourier_library(n_state_vars, n_frequencies, include_sin=True, include_cos=True):
    def make_sin_function(freqs):
        # here freqs is a tuple of the frequencies of the state variables
        # but, at most one frequency can be non-zero
        # e.g. (2, 0) means sin(2 * x)
        return lambda x: np.sin(np.multiply(freqs, x))

    def make_cos_function(freqs):
        # here freqs is a tuple of the frequencies of the state variables
        # but, at most one frequency can be non-zero
        # e.g. (2, 0) means cos(2 * x)
        return lambda x: np.cos(np.multiply(freqs, x))
    
    library_functions = []
    library_names = []
    for freq in range(1, n_frequencies + 1):
        # NOTE: we do not include zero frequency terms
        for i in range(n_state_vars):
            # create a tuple of frequencies where the ith frequency is non-zero
            freqs = tuple([0 if j != i else freq for j in range(n_state_vars)])
            # add the sin and cos functions to the library
            if include_sin:
                library_functions.append(make_sin_function(freqs))
                library_names.append(f"sin({freq}*x{i})")
            if include_cos:
                library_functions.append(make_cos_function(freqs))
                library_names.append(f"cos({freq}*x{i})")

    return library_functions, library_names


class SINDyModel:
    def __init__(self, threshold, poly_order=5, n_frequencies=0, include_sin=True, include_cos=True):
        self.threshold = threshold
        self.poly_order = poly_order
        self.n_frequencies = n_frequencies
        self.include_sin = include_sin
        self.include_cos = include_cos
        self.Xi = None

    
    def fit(self, x, x_dot, max_itterations=20):
        # transpose the data to the form (rows, cols) == (n_samples, n_features)
        # NOTE: it is just easier to work with the data in this form here
        x = x.T
        x_dot = x_dot.T

        assert x.shape == x_dot.shape
        assert x.shape[0] > x.shape[1] # we need more samples than state variables

        # add functions to the feature library
        self.feature_library = []
        self.feature_library_names = []
        library_functions, library_names = make_poly_library(x.shape[1], self.poly_order)
        self.feature_library += library_functions
        self.feature_library_names += library_names
        library_functions, library_names = make_fourier_library(x.shape[1], self.n_frequencies, self.include_sin, self.include_cos)
        self.feature_library += library_functions
        self.feature_library_names += library_names

        # Theta_ps = ps.PolynomialLibrary(degree=5).fit(x).transform(x)
        Theta = np.column_stack([f(x) for f in self.feature_library])

        Xi = np.linalg.lstsq(Theta, x_dot, rcond=None)[0]

        # repeat STLSQ multiple times
        for tmp in range(max_itterations):
            # get indices of small coefficients
            small_inds = np.abs(Xi) < self.threshold

            # check if there are any small (but non-zero) coefficients
            # if there are any, we should keep going. if there are none, we can stop.
            keep_going = np.any(small_inds & (Xi != 0))
            if not keep_going:
                # print(f"stopped after {tmp} itterations")
                break

            # set the small coefficients to zero
            Xi[small_inds] = 0

            # compute least squares on each state variable on the remaining terms
            for i in range(x.shape[1]):
                # get the indices of the non-zero coefficients for this state variable
                big_inds = ~small_inds[:, i]
                # compute least squares
                Xi[big_inds, i] = np.linalg.lstsq(Theta[:, big_inds], x_dot[:, i], rcond=None)[0]
        else:
            raise RuntimeError("STLSQ did not converge in the maximum number of itterations")

        # save the Xi matrix
        self.Xi = Xi


    def predict(self, x):
        # NOTE: "x" will take the form (n_state_vars, n_samples)
        #       so, x.T will take the form (n_samples, n_state_vars)

        # predict the time derivatives of the state variables
        x = x.T
        Theta = np.column_stack([f(x) for f in self.feature_library])
        x_dot = Theta @ self.Xi
        return x_dot.T
    
    
    def predict_single(self, x):
        # NOTE: "x" will take the form (n_state_vars,)
        return self.predict(x.reshape(-1, 1)).flatten()
    

    def print(self, decimals=5):
        # TODO (very minor): work on formatting the output a little better

        # print the learned model
        for i in range(self.Xi.shape[1]):
            print(f"x{i + 1} dot = ", end="") # NOTE: we use "i + 1" because the state variables are 1-indexed

            for j in range(self.Xi.shape[0]):
                if self.Xi[j, i] != 0:
                    print("+ ", end="")
                    print(f"{self.Xi[j, i]:.{decimals}f} {self.feature_library_names[j]}", end=" ")
            print()
