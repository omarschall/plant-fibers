import numpy as np

def test_loop(plants, datasets, climbing_fibers):
    ### Calculate test performance --- ###

    N_test_plants = len(plants)
    processed_data = np.zeros((5, N_test_plants))

    # First test cerebellum
    W_h = np.random.normal(0, 1 / np.sqrt(3), (n_h, n_in))
    W_o = np.random.normal(0, 1 / np.sqrt(n_h), (1, n_h + 1))
    test_cerebellum = Cerebellum(W_h, W_o, tanh)

    for i_plant, plant_data in enumerate(zip(plants[:N_test_plants], datasets[:N_test_plants])):
        plant, data = plant_data
        W_h = np.random.normal(0, 1 / np.sqrt(3), (n_h, n_in))
        W_o = np.random.normal(0, 1 / np.sqrt(n_h), (1, n_h + 1))
        test_cerebellum = Cerebellum(W_h, W_o, tanh)
        outer_loop.test_CF(data, plant, test_cerebellum, inner_lr=params['inner_lr'],
                           train_monitors=['CF', 'CF_label', 'RL_solution'],
                           test_monitors=['x_label', 'x_f', 'u'],
                           exploration_noise=params['exploration_noise'])

        # Calculate test_loss
        test_loss = np.mean(np.square(outer_loop.test_mons['x_f'] - outer_loop.test_mons['x_label']))
        cf_sgd_corr = np.corrcoef(outer_loop.train_mons['CF'], outer_loop.train_mons['CF_label'])[0, 1]
        cf_rl_corr = np.corrcoef(outer_loop.train_mons['CF'], outer_loop.train_mons['RL_solution'])[0, 1]

        processed_data[0, i_plant] = test_loss
        processed_data[1, i_plant] = cf_sgd_corr
        processed_data[2, i_plant] = cf_rl_corr