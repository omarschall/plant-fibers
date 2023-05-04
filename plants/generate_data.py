import numpy as np

def generate_gaussian_random_data(N, mean=0, std=1):
    """Generate identically distributioned x_0 and x_label values
    from a standard normal distribtuion."""

    data = {'train': {'x_0': np.random.normal(mean, std, (N, 1)),
                      'x_label': np.random.normal(mean, std, (N, 1))},
            'test': {'x_0': np.random.normal(mean, std, (N, 1)),
                     'x_label': np.random.normal(mean, std, (N, 1))}}

    return data

def generate_uniform_random_data(N, low=-1, high=1):
    """Generate identically distributioned x_0 and x_label values
    from a standard normal distribtuion."""

    data = {'train': {'x_0': np.random.uniform(low, high, (N, 1)),
                      'x_label': np.random.uniform(low, high, (N, 1))},
            'test': {'x_0': np.random.uniform(low, high, (N, 1)),
                     'x_label': np.random.uniform(low, high, (N, 1))}}

    return data