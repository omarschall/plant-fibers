from .Inner_Loop import Inner_Loop
import numpy as np

class Outer_Loop:

    def __init__(self, cerebellum, climbing_fibers, plants, reset_output=True,
                reset_kernel=True):
        """Initialize with an example cerebellum, the climbing fibers to be trained
        over the outer loop, and the family of plants to train on."""

        self.cerebellum = cerebellum
        self.n_in = cerebellum.n_in
        self.n_h = cerebellum.n_h
        self.climbing_fibers = climbing_fibers
        self.plants = plants
        self.reset_output = reset_output
        self.reset_kernel = reset_kernel

    def run(self, datasets, inner_lr, outer_lr, N_epochs=1, mode='train_CF', monitors=[],
            exploration_noise=0.01):
        """Run the outer loop over a list of datasets, with the same length
        as the number of plants, over a given number of epochs."""

        self.mons = {k: [] for k in monitors}

        for i_epoch in range(N_epochs):
            for i_data, data in enumerate(datasets):

                if self.reset_output:
                    self.reset_cerebellum()

                plant = self.plants[i_data]
                inner_loop = Inner_Loop(self.cerebellum, self.climbing_fibers,
                                        plant, inner_lr=inner_lr, outer_lr=outer_lr)
                inner_loop.run(data, mode=mode, monitors=monitors, verbose=False,
                               exploration_noise=exploration_noise)

                if (i_data / len(datasets) * 100) % 10 == 0:
                    print(i_data)

                if i_data == 0:
                    self.mons.update(inner_loop.mons)
                else:
                    for k in monitors:
                        self.mons[k] = np.concatenate([self.mons[k], inner_loop.mons[k]], axis=0)

    def reset_cerebellum(self):
        """Reset the cerebellum in the outer loop."""

        W_o = np.random.normal(0, 1 / np.sqrt(self.n_h), (1, self.n_h + 1))
        if self.reset_kernel:
            W_h = np.random.normal(0, 1 / np.sqrt(self.n_in), (self.n_h, self.n_in))
        else:
            W_h = self.cerebellum.W_h
        self.cerebellum.__init__(W_h, W_o, self.cerebellum.activation,
                                 tuning=self.cerebellum.tuning)

    def test_CF(self, data, plant, test_cerebellum, inner_lr, exploration_noise,
                train_monitors, test_monitors):
        """Test the climbing fibers' learning performance on a test plant for a
        freshly initialized test cerebellum."""

        ### Train our "test" cerebellum
        self.test_cerebellum = test_cerebellum
        inner_loop = Inner_Loop(self.test_cerebellum, self.climbing_fibers, plant,
                                inner_lr=inner_lr, outer_lr=0)
        inner_loop.run(data, mode='test_CF', monitors=train_monitors, verbose=False,
                       exploration_noise=exploration_noise)
        self.train_mons = inner_loop.mons.copy()

        ### Test our "test" cerebellum
        inner_loop = Inner_Loop(self.test_cerebellum, self.climbing_fibers, plant,
                                inner_lr=0, outer_lr=0)
        inner_loop.run(data, mode='test_CB', monitors=test_monitors, verbose=False)
        self.test_mons = inner_loop.mons.copy()

    def test_GD(self, data, plant, test_cerebellum, inner_lr, exploration_noise,
                train_monitors, test_monitors):
        """Mirroring the test_CF code, get baseline for using gradient descent."""

        ### Train our "test" cerebellum
        self.test_cerebellum = test_cerebellum
        inner_loop = Inner_Loop(self.test_cerebellum, self.climbing_fibers, plant,
                                inner_lr=inner_lr, outer_lr=0)
        inner_loop.run(data, mode='test_CF', monitors=train_monitors, verbose=False,
                       use_GD=True, exploration_noise=exploration_noise)
        self.train_mons = inner_loop.mons.copy()

        ### Test our "test" cerebellum
        inner_loop = Inner_Loop(self.test_cerebellum, self.climbing_fibers, plant,
                                inner_lr=0, outer_lr=0)
        inner_loop.run(data, mode='test_CB', monitors=test_monitors, verbose=False)
        self.test_mons = inner_loop.mons.copy()

    def test_RL(self, data, plant, test_cerebellum, inner_lr,
                train_monitors, test_monitors, exploration_noise):
        """Mirroring the test_CF code, get baseline for using reinforcement
        learing."""

        ### Train our "test" cerebellum
        self.test_cerebellum = test_cerebellum
        inner_loop = Inner_Loop(self.test_cerebellum, self.climbing_fibers, plant,
                                inner_lr=inner_lr, outer_lr=0)
        inner_loop.run(data, mode='test_CF', monitors=train_monitors, verbose=False,
                       use_RL=True, exploration_noise=exploration_noise)
        self.train_mons = inner_loop.mons.copy()

        ### Test our "test" cerebellum
        inner_loop = Inner_Loop(self.test_cerebellum, self.climbing_fibers, plant,
                                inner_lr=0, outer_lr=0)
        inner_loop.run(data, mode='test_CB', monitors=test_monitors, verbose=False)
        self.test_mons = inner_loop.mons.copy()

    def test_on_plant_family(self, datasets, plants, exploration_noise, inner_lr):
        """Run test of the climbing fibers and the baselines for a family of
        plants."""

        processed_data = np.zeros((5, len(plants)))
        for i_plant, plant_data in enumerate(zip(plants, datasets)):
            plant, data = plant_data
            self.reset_cerebellum()
            self.test_CF(data, plant, self.cerebellum, inner_lr=inner_lr,
                         train_monitors=['CF', 'CF_label', 'RL_solution'],
                         test_monitors=['x_label', 'x_f'],
                         exploration_noise=exploration_noise)

            # Calculate test_loss
            test_loss = np.mean(np.square(self.test_mons['x_f'] - self.test_mons['x_label']))
            cf_sgd_corr = np.corrcoef(self.train_mons['CF'], self.train_mons['CF_label'])[0, 1]
            cf_rl_corr = np.corrcoef(self.train_mons['CF'], self.train_mons['RL_solution'])[0, 1]

            self.test_GD(data, plant, self.cerebellum, inner_lr=inner_lr,
                         train_monitors=[],
                         test_monitors=['x_label', 'x_f'],
                         exploration_noise=exploration_noise)
            gd_loss = np.mean(np.square(self.test_mons['x_f'] - self.test_mons['x_label']))

            self.test_RL(data, plant, self.cerebellum, inner_lr=inner_lr,
                         train_monitors=[],
                         test_monitors=['x_label', 'x_f'],
                         exploration_noise=exploration_noise)
            rl_loss = np.mean(np.square(self.test_mons['x_f'] - self.test_mons['x_label']))

            processed_data[0, i_plant] = test_loss
            processed_data[1, i_plant] = gd_loss
            processed_data[2, i_plant] = rl_loss
            processed_data[3, i_plant] = cf_sgd_corr
            processed_data[4, i_plant] = cf_rl_corr

            return processed_data