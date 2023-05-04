from .Inner_Loop import Inner_Loop
import numpy as np

class Outer_Loop:

    def __init__(self, cerebellum, climbing_fibers, plants):

        self.cerebellum = cerebellum
        self.n_in = cerebellum.n_in
        self.n_h = cerebellum.n_h
        self.climbing_fibers = climbing_fibers
        self.plants = plants

    def run(self, datasets, inner_lr, outer_lr, N_epochs=1, mode='train_CF', monitors=[],
            reset_cerebellum=True, reset_kernel=True, exploration_noise=0.01):

        self.mons = {k: [] for k in monitors}

        for i_epoch in range(N_epochs):
            for i_data, data in enumerate(datasets):

                if reset_cerebellum:
                    W_o = np.random.normal(0, 1 / np.sqrt(self.n_h), (1, self.n_h + 1))
                    if reset_kernel:
                        W_h = np.random.normal(0, 1 / np.sqrt(2), (self.n_h, self.n_in))
                    else:
                        W_h = self.cerebellum.W_h
                    self.cerebellum.__init__(W_h, W_o, self.cerebellum.activation)

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

    def test_CF(self, data, plant, test_cerebellum, inner_lr, exploration_noise,
                train_monitors, test_monitors):

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

    def test_GD(self, data, plant, test_cerebellum, inner_lr,
                train_monitors, test_monitors):

        ### Train our "test" cerebellum
        self.test_cerebellum = test_cerebellum
        inner_loop = Inner_Loop(self.test_cerebellum, self.climbing_fibers, plant,
                                inner_lr=inner_lr, outer_lr=0)
        inner_loop.run(data, mode='test_CF', monitors=train_monitors, verbose=False,
                       use_GD=True)
        self.train_mons = inner_loop.mons.copy()

        ### Test our "test" cerebellum
        inner_loop = Inner_Loop(self.test_cerebellum, self.climbing_fibers, plant,
                                inner_lr=0, outer_lr=0)
        inner_loop.run(data, mode='test_CB', monitors=test_monitors, verbose=False)
        self.test_mons = inner_loop.mons.copy()

    def test_RL(self, data, plant, test_cerebellum, inner_lr,
                train_monitors, test_monitors, exploration_noise):

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