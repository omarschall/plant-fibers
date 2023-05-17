import numpy as np

class Climbing_Fibers_1HLMLP:
    """Parameterized by theta = (W_1, W_2), produces training signals
    for the cerebellum. Currently hard-coded as 1 hidden layer MLP."""

    def __init__(self, W_1, W_2, activation, final_output, spoonfeed_RL=0,
                 RL_pieces_scale=1, tuning=None):
        """Initialize with the initial values of the input-hidden (W_1)
        and hidden-output (W_2) weights."""

        self.W_1 = W_1
        self.W_2 = W_2
        self.activation = activation
        self.final_output = final_output
        self.spoonfeed_RL = spoonfeed_RL
        self.RL_pieces_scale = RL_pieces_scale
        self.tuning = tuning

        assert W_1.shape[0] == W_2.shape[0] - 1

        self.n_h = W_1.shape[1]

    def __call__(self, x_0, x_label, x_f, u, noise, R, R_avg, exploration_noise=0.01):
        """Return a CF output given the information fed in"""

        #Rescale RL solution by spoonfeed factor
        RL_solution = self.spoonfeed_RL * noise * (R - R_avg)

        #Rescale pieces of RL solution by RL pieces factor
        noise *= self.RL_pieces_scale
        R *= self.RL_pieces_scale
        R_avg *= self.RL_pieces_scale

        #z-score noise
        if exploration_noise > 0:
            noise = noise / exploration_noise

        #Concatenate inputs to climbing fibers
        self.x_cf = np.concatenate([x_0, x_label, x_f, u,
                                    noise, R, R_avg, RL_solution, np.array([1])])

        #Place all values into turning curves
        if self.tuning is not None:
            self.x_cf = np.concatenate([self.tuning(self.x_cf[:-1]), np.array([1])])

        #Compute forward pass of the CF system
        self.h = self.W_1.dot(self.x_cf)
        self.a = self.activation.f(self.h)
        self.a_hat = np.concatenate([self.a, np.array([1])])
        self.pre_output = self.W_2.dot(self.a_hat)
        self.CF = self.final_output.f(self.pre_output)

        return self.CF.copy()

    def get_dPsi_dW(self):
        """Calculate the derivative of the output with respect to the parameters
        theta, as a list of two arrays of shape """

        self.dPsi_dpre = self.final_output.f_prime(self.pre_output)
        self.dPsi_dW_2 = self.dPsi_dpre * self.a_hat
        self.dPsi_dW_1 = self.dPsi_dpre * np.multiply.outer(self.W_2[:-1] * self.activation.f_prime(self.h),
                                                            self.x_cf)

        return [self.dPsi_dW_1.copy(), self.dPsi_dW_2.copy()]

    def update_theta(self, error, plant_derivative, phi, phi_test, dPsi_dW, outer_lr):
        """Update CF params based on the performance error, backpropagated to
        each layer."""

        dL_dPsi = np.sum(error * plant_derivative * phi * phi_test)

        self.dL_dW_1 = dL_dPsi * dPsi_dW[0]
        self.dL_dW_2 = dL_dPsi * dPsi_dW[1]

        self.W_1 = np.squeeze(self.W_1 - outer_lr * self.dL_dW_1)
        self.W_2 = np.squeeze(self.W_2 - outer_lr * self.dL_dW_2)

    def update_theta_on_gradient(self, CF_error, dPsi_dW, outer_lr):
        """Update CF params based on matching a given error signal CF_error,
        e.g. the difference between the CF output and gradient descent soln."""

        self.W_1 = np.squeeze(self.W_1 - outer_lr * CF_error * dPsi_dW[0])
        self.W_2 = np.squeeze(self.W_2 - outer_lr * CF_error * dPsi_dW[1])
