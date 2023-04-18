from .Inner_Loop import Inner_Loop
import numpy as np
from cerebellum import Cerebellum

class Outer_Loop:

    def __init__(self, cerebellum, climbing_fibers, plants):

        self.cerebellum = cerebellum
        self.n_in = cerebellum.n_in
        self.n_h = cerebellum.n_h
        self.climbing_fibers = climbing_fibers
        self.plants = plants

    def run(self, datasets, inner_lr, outer_lr, mode='train_CF', monitors=[], reset_cerebellum=False):

        self.mons = {k: [] for k in monitors}

        for i_data, data in enumerate(datasets):

            if reset_cerebellum:
                W_h = np.random.normal(0, 1 / np.sqrt(2), (self.n_h, self.n_in))
                W_o = np.random.normal(0, 1 / np.sqrt(self.n_h), (1, self.n_h + 1))
                self.cerebellum = Cerebellum(W_h, W_o, self.cerebellum.activation)

            plant = self.plants[i_data]
            inner_loop = Inner_Loop(self.cerebellum, self.climbing_fibers,
                                    plant, inner_lr=inner_lr, outer_lr=outer_lr)
            inner_loop.run(data, mode=mode, monitors=monitors, verbose=False)

            if (i_data / len(datasets) * 100) % 10 == 0:
                print(i_data)

            if i_data == 0:
                self.mons.update(inner_loop.mons)
            else:
                for k in monitors:
                    self.mons[k] = np.concatenate([self.mons[k], inner_loop.mons[k]], axis=0)