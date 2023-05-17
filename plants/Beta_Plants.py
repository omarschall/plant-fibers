from .generate_data import generate_uniform_random_data
from .GP_Plants import plant_from_GP, plant_derivative_from_GP
import numpy as np
from functions import *
from functools import partial
from functions import Function
from scipy.stats import beta

def generate_beta_plants_and_datasets(N_plants, N_inner, N_u_discrete,
                                      u_min=-1, u_max=1, gamma=1):
    """Generate the 'onion' family of plants by beta-CDF-like functions and
    varying the concavity and the sign of the slopes."""

    u_range = np.linspace(u_min, u_max, N_u_discrete)
    plants = []
    datasets = []
    for i_plant in range(N_plants):

        #concavity = np.random.uniform(-concavity_max, concavity_max)
        concavity = np.random.normal(0, 2*gamma)
        a = 1 + relu.f(concavity)
        b = 1 + relu.f(-concavity)
        GP = 2 * beta.cdf((u_range + 1) / 2, a=a, b=b) - 1
        slope = np.random.choice([-1, 1])
        GP *= slope
        plants.append(Function(partial(plant_from_GP, x_GP=u_range, GP=GP),
                               partial(plant_derivative_from_GP, x_GP=u_range, GP=GP)))
        datasets.append(generate_uniform_random_data(N_inner, -1, 1))

    return plants, datasets