import numpy as np

def generate_gaussian_random_data(N):
    """Generate identically distributioned x_0 and x_label values
    from a standard normal distribtuion."""

    data = {'train': {'x_0': np.random.normal(0, 1, (N, 1)),
                      'x_label': np.random.normal(0, 1, (N, 1))},
            'test': {'x_0': np.random.normal(0, 1, (N, 1)),
                     'x_label': np.random.normal(0, 1, (N, 1))}}

    return data