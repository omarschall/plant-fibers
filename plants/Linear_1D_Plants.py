import numpy as np
from functools import partial
from functions import *

def linear_plant(x_0, u, a):
    """Linear plant, where final state is linear function of the control, with
    slope parameterized by a. Ignores initial state."""

    return a * u


def affine_linear_plant(x_0, u, a):
    """Affine linear plant, where final state is linear function of the control,
    with slope parameterized by a. Initial state provides shift."""

    return x_0 + a * u

def linear_plant_derivative(x_0, u, a):
    """Derivative of the linear plant with respect to the control."""

    return a

def generate_linear_plants(N_plants, a_dist='normal'):
    """Generate a list of linear plants, with slopes a distributed either
    according to a binary -1 and 1, or from a normal distribution."""

    if a_dist == 'normal':
        a_values = np.random.normal(0, 1, N_plants)
    if a_dist == 'binary':
        a_values = np.random.choice([-1, 1], size=N_plants)

    plants = []
    for i_plant in range(N_plants):

        a = a_values[i_plant]
        plant = Function(partial(linear_plant, a=a),
                         partial(linear_plant_derivative, a=a))
        plants.append(plant)

    return plants

def generate_affine_linear_plants(N_plants, a_dist='normal'):
    """Generate a list of affine linear plants, with slopes a distributed either
    according to a binary -1 and 1, or from a normal distribution."""

    if a_dist == 'normal':
        a_values = np.random.normal(0, 1, N_plants)
    if a_dist == 'binary':
        a_values = np.random.choice([-1, 1], size=N_plants)

    plants = []
    for i_plant in range(N_plants):

        a = a_values[i_plant]
        plant = Function(partial(affine_linear_plant, a=a),
                         partial(linear_plant_derivative, a=a))
        plants.append(plant)

    return plants