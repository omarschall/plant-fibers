import matplotlib.pyplot as plt
import numpy as np

def plot_2d_array_of_config_results(configs_array, results_array, key_order,
                                    log_scale=False, tick_rounding=3, **imshow_kwargs):
    """Given an array of configs (must be 2D) and corresponding results as
    floats, plots the result in a 2D grid averaging over random seeds."""

    fig = plt.figure()

    plt.imshow(results_array.mean(-1), **imshow_kwargs)

    if log_scale:
        plt.yticks(range(results_array.shape[0]),
                   np.round(np.log10(configs_array[key_order[0]]),
                            tick_rounding))
        plt.xticks(range(results_array.shape[1]),
                   np.round(np.log10(configs_array[key_order[1]]),
                            tick_rounding))
    else:
        plt.yticks(range(results_array.shape[0]),
                   np.round(configs_array[key_order[0]],
                            tick_rounding))
        plt.xticks(range(results_array.shape[1]),
                   np.round(configs_array[key_order[1]],
                            tick_rounding))

    plt.ylabel(key_order[0])
    plt.xlabel(key_order[1])
    plt.colorbar()

    return fig

def plot_test_plant_results(outer_loop, plants, datasets, exploration_noise,
                            inner_lr, mode='CF'):

    n_plants = len(plants)
    n_rows = int(np.ceil(np.sqrt(n_plants)))
    fig, ax = plt.subplots(n_rows, n_rows, figsize=(12, 12))
    for i_plant, plant_data in enumerate(zip(plants, datasets)):
        plant, data = plant_data
        outer_loop.reset_cerebellum()
        test_cerebellum = outer_loop.cerebellum
        if mode == 'CF':
            outer_loop.test_CF(data, plant, test_cerebellum, inner_lr=inner_lr,
                               train_monitors=[],
                               test_monitors=['x_label', 'x_f', 'u'],
                               exploration_noise=exploration_noise)
        if mode == 'RL':
            outer_loop.test_RL(data, plant, test_cerebellum, inner_lr=inner_lr,
                               train_monitors=[],
                               test_monitors=['x_label', 'x_f', 'u'],
                               exploration_noise=exploration_noise)
        if mode == 'GD':
            outer_loop.test_GD(data, plant, test_cerebellum, inner_lr=inner_lr,
                               train_monitors=[],
                               test_monitors=['x_label', 'x_f', 'u'],
                               exploration_noise=exploration_noise)

        i_x = i_plant // n_rows
        i_y = i_plant % n_rows
        ax[i_x, i_y].plot(outer_loop.test_mons['u'], outer_loop.test_mons['x_label'], '.')
        u = np.arange(-1, 1, 0.01)
        ax[i_x, i_y].plot(u, plant.f(0, u), 'r')
        test_loss = np.mean(np.square(outer_loop.test_mons['x_f'] - outer_loop.test_mons['x_label']))
        ax[i_x, i_y].set_title('loss = {0:.2f}'.format(test_loss))
        ax[i_x, i_y].set_xlim([-1,1])
        ax[i_x, i_y].set_ylim([-1,1])
        ax[i_x, i_y].set_xticks([])
        ax[i_x, i_y].set_yticks([])
    ax[0, 0].set_xlabel('$u$')
    ax[0, 0].set_ylabel('$x_f$')

    return fig