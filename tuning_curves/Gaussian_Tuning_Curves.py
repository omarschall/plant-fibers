from scipy.stats import norm

class Gaussian_Tuning_Curves:
    """Instance of gaussian tuning curves"""

    def __init__(self, means, stds):
        """Initialize by specifying the means and the stds of the
        turning curves. Must have same shape of (n)."""

        self.means = means
        self.stds = stds

    def __call__(self, x):
        """For a given vector, represents each float in the vector
        as a list of tuning curve activations, then concatenates the results."""

        return norm.pdf(x.reshape(-1, 1), loc=self.means, scale=self.stds).reshape(-1)