from src.SINDyModel import *

class ESINDyModel:
    def __init__(self, n_state_vars, threshold, inclusion_threshold=0.6, poly_order=5, n_frequencies=0):
        self.n_state_vars = n_state_vars
        self.threshold = threshold
        self.inclusion_threshold = inclusion_threshold
        self.poly_order = poly_order
        self.n_freq = n_frequencies
    
    def fit(self, x, x_dot, n_bootstraps=100, seed=0):
        n_samples = x.shape[1]
        
        Xis = []

        np.random.seed(seed)

        for _ in range(n_bootstraps):
            # randomly select a subset of the data
            indices = np.random.choice(n_samples, n_samples, replace=True)
            x_subset = x[:, indices]
            x_dot_subset = x_dot[:, indices]

            # create the SINDy model
            sindy_model = SINDyModel(self.n_state_vars, self.threshold, self.poly_order, self.n_freq)
            sindy_model.fit(x_subset, x_dot_subset)

            Xis.append(sindy_model.Xi)
        
        # convert the list of Xi arrays to a 3D array
        # it's shape will be (n_bootstraps, n_features, n_state_vars)
        Xis = np.array(Xis)

        ip = np.mean(Xis != 0, axis=0)
        
        Xi_agg = np.median(Xis, axis=0)

        # set the Xi values to zero if the inclusion probability is below the threshold
        Xi_agg[ip < self.inclusion_threshold] = 0

        # create the median model and save it to this object
        self.agg_model = SINDyModel(self.n_state_vars, self.threshold, self.poly_order, self.n_freq)
        self.agg_model.Xi = Xi_agg

        # this is not used in the class, but is useful in testing
        self.Xi = Xi_agg
    

    def predict(self, x):
        return self.agg_model.predict(x)
    
    def predict_single(self, x):
        return self.agg_model.predict_single(x)

    def print(self, decimals=5):
        self.agg_model.print(decimals)
