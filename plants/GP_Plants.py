from .generate_data import generate_gaussian_random_data
import numpy as np
from functools import partial
from functions import Function

def plant_from_GP(x_0, u, x_GP, GP):
    """Plant function for a plant parameterized by a Gaussian process."""

    return np.interp(u, x_GP, GP)

def plant_derivative_from_GP(x_0, u, x_GP, GP):
    """Plant derivative for a given Gaussian process (not the derivative)."""

    GP_prime = (GP[1:] - GP[:-1]) / (x_GP[1:] - x_GP[:-1])

    return np.interp(u, x_GP[:-1], GP_prime)

def generate_GP_plants_and_datasets(N_plants, N_inner, lambda_GP=2.0, tau_GP=0.9,
                                    N_u_discrete=100, u_min=-5, u_max=5):

    u_range = np.linspace(u_min, u_max, N_u_discrete)

    # build kernel discretization
    C0 = np.zeros((N_u_discrete, N_u_discrete))
    for i in range(N_u_discrete):
        for j in range(N_u_discrete):
            C0[i, j] = np.exp(-1 / lambda_GP**2 * (u_range[i] - u_range[j])**2)

    # combine kernels into a multi-process covariance matrix
    C = np.zeros((N_u_discrete * N_plants, N_u_discrete * N_plants))
    for b in range(N_plants):
        for b2 in range(N_plants):
            if b == b2:
                C[b * N_u_discrete:(b + 1) * N_u_discrete,
                  b * N_u_discrete:(b + 1) * N_u_discrete] = C0[:, :]
            else:
                C[b * N_u_discrete:(b + 1) * N_u_discrete,
                  b2 * N_u_discrete:(b2 + 1) * N_u_discrete] = tau_GP * C0[:, :]

    # sample the multi-process and reshape into multiple GP samples
    X = np.random.multivariate_normal(np.zeros(N_u_discrete * N_plants), C, 1).reshape(N_plants, N_u_discrete)

    plants = []
    datasets = []
    for i_plant in range(N_plants):

        GP = X[i_plant]
        plants.append(Function(partial(plant_from_GP, x_GP=u_range, GP=GP),
                               partial(plant_derivative_from_GP, x_GP=u_range, GP=GP)))

        #Compute reasonable data range
        GP_min = np.amin(GP)
        GP_max = np.amax(GP)
        mean = (GP_max - GP_min)/2 + GP_min
        std = np.abs(GP_max - GP_min) / 6
        datasets.append(generate_gaussian_random_data(N_inner, mean=mean, std=std))

    return plants, datasets