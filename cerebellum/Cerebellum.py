import numpy as np

class Cerebellum:
    """The forward model, which produces a control signal in the
    cerebellar nuclei."""

    def __init__(self, W_h, W_o, activation):
        """Initialize instance with (fixed) W_h weights for the expansion
        to the GC layer, (trainable) weights W_o for the GC to control
        signal, and elementwise nonlinearity."""

        self.W_h = W_h
        self.W_o = W_o
        self.activation = activation
        self.n_in = W_h.shape[1]
        self.n_h = W_h.shape[0]

        # right-handed matrix-vector multiplication
        assert self.W_h.shape[0] == self.W_o.shape[1] - 1

        # Initial state values
        self.phi = np.zeros(self.W_h.shape[1])

    def forward_pass(self, x_0, x_label, exploration_noise=0.01):
        """Takes in an input x_0, a desired final state x_label, concatenates
        them, and maps them forward, eventually returning a value of u."""

        # Concatenate initial state with desired final state
        x = np.concatenate([x_0, x_label])

        # GC activation
        self.phi = self.activation.f(np.dot(self.W_h, x))
        self.phi_hat = np.concatenate([self.phi, np.array([1])])

        # Control signal output
        self.noise = np.random.normal(0, exploration_noise, (1))
        self.u = - self.W_o.dot(self.phi_hat) + self.noise

        return self.u.copy()