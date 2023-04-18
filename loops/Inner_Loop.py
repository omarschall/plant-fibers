import numpy as np
import time
from utils import *

class Inner_Loop:

    def __init__(self, cerebellum, climbing_fibers, plant, lambda_R=2,
                 alpha_avg=0.05, inner_lr=0.01, outer_lr=0.001):
        """Initialize an instance of inner loop training with
        the current neural architectures.

        kwargs:
            alpha_avg (float): inverse time constant between 0 and
                1 specifying the time scale of averaging u and R.
            lambda_R (float): spatial scale of reward as function
                of error.
            inner_lr (float): learning rate of cerebellum parameters in
                inner loop.
            outer_lr (float): learning rate of climbing fiber parameters
                in the outer loop."""

        self.cerebellum = cerebellum
        self.climbing_fibers = climbing_fibers
        self.plant = plant

        self.alpha_avg = alpha_avg
        self.lambda_R = lambda_R
        self.R_avg = np.array([0])
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr

    def run(self, data, mode='train_CF', monitors=[], verbose=True):
        """Take in a data dict of x_0 values and x_label values"""

        # Assess size of data
        self.N_loop = len(data['train']['x_0'])
        self.report_interval = max(self.N_loop // 10, 1)

        # Initialize monitors
        self.mons = {k: [] for k in monitors}

        # Initialize start time
        self.start_time = time.time()

        for i_x in range(self.N_loop):

            self.i_x = i_x

            self.x_0 = data['train']['x_0'][i_x]
            self.x_label = data['train']['x_label'][i_x]

            # Compute cerebellum output, final state, and reward
            self.u = self.cerebellum.forward_pass(self.x_0, self.x_label)
            self.phi = self.cerebellum.phi.copy()
            self.x_f = self.plant.f(self.x_0, self.u)
            self.train_error = self.x_f - self.x_label
            self.R = np.exp(-1 / self.lambda_R * np.abs(self.x_label - self.x_f))

            # Update average reward and control
            self.R_avg = (1 - self.alpha_avg) * self.R_avg + self.alpha_avg * self.R

            # Calculate CF output
            self.CF = self.climbing_fibers(self.x_0, self.x_label, self.x_f,
                                           self.u, self.cerebellum.noise,
                                           self.R, self.R_avg)
            self.CF_label = -self.x_f / self.u * (self.x_f - self.x_label)
            self.RL_solution = self.cerebellum.noise * (self.R - self.R_avg)

            # Update cerebellum parameters
            self.cerebellum.W_o = self.cerebellum.W_o - self.inner_lr * self.CF * self.cerebellum.phi_hat

            if mode == 'train_CF':
                # Update CF parameters
                self.x_0_test = data['test']['x_0'][i_x]
                self.x_label_test = data['test']['x_label'][i_x]
                self.u_test = self.cerebellum.forward_pass(self.x_0_test, self.x_label_test)
                self.phi_test = self.cerebellum.phi.copy()
                self.x_f_test = self.plant.f(self.x_0_test, self.u_test)
                self.error = self.x_f_test - self.x_label_test
                self.g_prime = self.plant.f_prime(self.x_0_test, self.u_test)
                dPsi_dW = self.climbing_fibers.get_dPsi_dW()
                self.climbing_fibers.update_theta(self.error, self.g_prime, self.phi, self.phi_test, dPsi_dW,
                                                  self.outer_lr)

            # Update monitors
            self.update_monitors()
            self.get_radii_and_norms()

            # Make report if conditions are met
            if (self.i_x % self.report_interval == 0 and
                    self.i_x > 0 and verbose):
                self.report_progress()

        self.monitors_to_arrays()

    def update_monitors(self):
        """Loops through the monitor keys and appends current value of any
        object's attribute found."""

        for key in self.mons:
            try:
                self.mons[key].append(rgetattr(self, key))
            except AttributeError:
                pass

    def get_radii_and_norms(self):
        """Calculates the spectral radii and/or norms of any monitor keys
        where this is specified."""

        for feature, func in zip(['radius', 'norm'],
                                 [get_spectral_radius, norm]):
            for key in self.mons:
                if feature in key:
                    attr = key.split('-')[0]
                    self.mons[key].append(func(rgetattr(self, attr)))

    def report_progress(self):
        """"Reports progress at specified interval, including test run
        performance if specified."""

        progress = np.round((self.i_x / self.N_loop) * 100, 2)
        time_elapsed = np.round(time.time() - self.start_time, 1)

        summary = '\rProgress: {}% complete \nTime Elapsed: {}s \n'
        print(summary.format(progress, time_elapsed))

    def monitors_to_arrays(self):
        """Recasts monitors (lists by default) as numpy arrays for ease of use
        after running."""

        for key in self.mons:
            try:
                self.mons[key] = np.array(self.mons[key])
            except ValueError:
                pass