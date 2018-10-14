"""Implementation of the MLP algorithm and the IBV problem.

MLP - multi-layer-perceptron
IBV - initial/boundary value
"""
from abc import ABC, abstractmethod
import os
from sys import exit
import autograd.numpy as np
from autograd import grad
from autograd import elementwise_grad as egrad
from random import shuffle


class IBVProblem(ABC):
    """Wrapper class to solve an initial/boundary value problem."""

    def __init__(self, name, mlp):
        """Initialize initial/boundary value problem."""
        self.name = name
        self.mlp = mlp
        self.control_points = []
        self.loss_control_points = []
        self.loss_indices = []
        self.losses = []
        self.every = 1

    @abstractmethod
    def loss_function_kernel(self, weights, x_in, point_index):
        """Return loss function kernel."""
        pass

    @abstractmethod
    def trial_function(self, x_in):
        """Return trial function."""
        pass

    def set_control_points(self, points):
        """Set the control points externally."""
        self.control_points = points
        self.loss_control_points = points
        self.loss_indices = range(len(points))

    def set_loss_control_points(self):
        """Select a subset of control points used for training."""
        training_points = []
        training_indices = []
        points_and_indices = [(i, point) for i, point in
                              enumerate(self.control_points)]
        shuffle(points_and_indices)
        for i in range(len(points_and_indices)):
            if (i + 1) % self.every == 0:
                training_points.append(points_and_indices[i][1])
                training_indices.append(points_and_indices[i][0])
        self.loss_control_points = np.asarray(training_points)
        self.loss_indices = np.asarray(training_indices)
        len_training = len(self.loss_control_points)
        len_all = len(self.control_points)
        print("Selecting subset for training: {} out of {} \
selected.".format(len_training, len_all))

    def place_control_points(self, method=[], domain=[],
                             step_size=[], growths=[]):
        """Places control points within a given domain."""
        [self.mlp.assert_input_length(input_list) for input_list in
            [method, domain, step_size, growths]]

        print("\nPlacing control points for {} input variables."
              .format(len(domain)))
        print("------------------------------------------------")

        current_var = [domain[i][0] + step_size[i] for i in range(len(domain))]
        points = [[domain[i][0]] for i in range(len(domain))]
        lengths = []

        for i, method_i in enumerate(method):
            if method_i == 'uniform':
                while (current_var[i] + 1.0E-6) < domain[i][1]:
                    points[i].append(current_var[i])
                    current_var[i] += step_size[i]
            else:
                while (current_var[i] + 1.0E-6) < domain[i][1]:
                    points[i].append(current_var[i])
                    current_var[i] *= growths[i]
            points[i].append(domain[i][1])
            lengths.append(len(points[i]))
        total = np.product(np.asarray(lengths))
        all_points = []
        assert self.mlp.number_of_inputs <= 3
        if self.mlp.number_of_inputs == 1:
            all_points = np.asarray(points[0])
        elif self.mlp.number_of_inputs == 2:
            for p0 in points[0]:
                for p1 in points[1]:
                    all_points.append(np.asarray([p0, p1]))
        elif self.mlp.number_of_inputs == 3:
            for p0 in points[0]:
                for p1 in points[1]:
                    for p2 in points[2]:
                        all_points.append(np.asarray([p0, p1, p2]))
        self.control_points = np.asarray(all_points)
        self.loss_control_points = self.control_points
        self.loss_indices = range(len(self.control_points))

        print("Control points per input parameter: {}".format(lengths))
        print("The total number of control points is {}".format(total))

    def solve(self, max_iter=1000, learning_rate=0.01, verbose=100,
              tolerance=0.1, adaptive=False, every=1, reduction=0.9):
        """Solve the minimization problem."""
        if adaptive and every > 1:
            self.every = every
            self.set_loss_control_points()

        initial_loss = self.loss_function()
        print("\nThe initial loss is {}\n".format(initial_loss))
        self.losses.append(initial_loss)

        m_t = 0.0 * self.mlp.get_weights()
        v_t = 0.0 * self.mlp.get_weights()
        beta_1 = 0.9
        beta_2 = 0.999
        eps = 1.0E-8
        min_loss = 1.0E10
        iter_adaptive = 0

        for i in range(1, max_iter+1):
            grad_E = self.loss_function_gradient()
            m_t = beta_1 * m_t + (1.0 - beta_1) * grad_E
            v_t = beta_2 * v_t + (1.0 - beta_2) * np.square(grad_E)
            m_th = m_t / (1.0 - beta_1**i)
            v_th = v_t / (1.0 - beta_2**i)
            update = -learning_rate * m_th / (np.sqrt(v_th) + eps)
            self.mlp.update_weights(update)
            iter_adaptive += 1
            if i % verbose == 0:
                current_loss = self.loss_function()
                self.losses.append(current_loss)
                if current_loss < min_loss:
                    self.mlp.write_weights_to_disk(self.name)
                    min_loss = current_loss
                    print("The loss after {} iterations is E={}"
                          .format(i, current_loss))
                if current_loss < tolerance:
                    iter_adaptive = 0
                    if adaptive and self.every > 1:
                        self.every = int(self.every * reduction)
                        print("Using every {}th point for \
                        training.".format(self.every))
                        self.set_loss_control_points()
                        min_loss = 1.0E10
                    else:
                        print("Tolerance reached after {} \
                        iterations.".format(i))
                        exit()
                elif adaptive and iter_adaptive > 1000:
                    print("Re-setting points: using every {}th point for \
training.".format(self.every))
                    self.set_loss_control_points()
                    iter_adaptive = 0

    def loss_function(self):
        """Compute the loss function based on control points."""
        E = 0.0
        for ind, x_in in enumerate(self.loss_control_points):
            E += self.loss_function_kernel(self.mlp.weights, x_in,
                                           self.loss_indices[ind])
        return E

    def loss_function_gradient(self):
        """Compute the gradient of the loss function."""
        def local_loss(weights):
            E = 0.0
            for ind, x_in in enumerate(self.loss_control_points):
                E += self.loss_function_kernel(weights, x_in,
                                               self.loss_indices[ind])
            return E
        return grad(local_loss)(self.mlp.weights)

    def predict(self, points):
        """Compute the trial function for given input points."""
        result = np.zeros(len(points))
        for index, point in enumerate(points):
            result[index] = self.trial_function(point, self.mlp.weights)
        return result

    def l2_norm(self, points, reference):
        """Compute the L2 norm."""
        prediction = self.predict(points)
        return np.sqrt(np.sum(np.square(prediction - reference))) \
            / len(reference)

    def lmax_norm(self, points, reference):
        """Compute the maximum norm."""
        prediction = self.predict(points)
        return np.max(np.abs(prediction - reference))

    def get_losses(self):
        """Return the losses."""
        return self.losses


class SimpleMLP:
    """Implemnts a simple multi layer percetron algorithm."""

    def __init__(self, name, number_of_inputs, neurons_per_layer,
                 hidden_layers, initialization='xavier_he'):
        """Initialize MLP."""
        self.name = name
        self.number_of_inputs = number_of_inputs
        self.neurons_per_layer = neurons_per_layer
        self.hidden_layers = hidden_layers
        self.weights = self.initialize_weights(initialization)

    def initialize_weights(self, method):
        """Intialize weights and biases.

            Biases are always initialized with zero.
            The weights can be chosen randomly or with the Xavier-He
            initialization. The MLP has one input layer, at least one
            hidden layers, and one output layer. All layers except for
            the output have exactly one bias unit. All biases and weights
            (input and output) are stored in the array 'weights'.

        Parameters
        ----------
        method - string : Defines the initialization method;
                          Can be either 'random' or 'xavier_he'

        Set/Return
        ----------
        weights - array-like : 2D array containing all weights and biases;
                               It has the shape [rows, columns] with
                               rows = rows_input + rows_hidden + rows_output
                               (further explanation in the code)
                               columns = neurons_per_layer (all hidden layers
                               have the same number of neurons)
                               [0, :] accesses the input bias
                               [1, :] accesses the weights between the first
                               input neuron and all neurons of the first hidden
                               layer
                               ...
                               [number_of_inputs + 1, :] accesses the bias of
                               the first hidden layer
                               [number_of_inputs + 2, :] accesses the weights
                               between the first neuron of the first hidden
                               layer and all neurons of the second hidden layer
                               ...
                               [(number_of_inputs + 1) * (layer - 1), :]
                               accesses the bias of 'layer' with layer=1 for
                               the first hidden layer and so on

        """
        # each input is connected to each neuron in the first hidden layer;
        # plus one for the bias
        input_rows = self.number_of_inputs + 1
        # each neuron in the hidden layer is connected to each neuron in the
        # next layer; plus one for the bias
        hidden_rows = self.neurons_per_layer + 1
        # +1 for the weights between last hidden layer and output neuron
        rows = input_rows + hidden_rows * (self.hidden_layers - 1) + 1
        weights = np.zeros([rows, self.neurons_per_layer]).astype(np.float64)
        np.random.seed(1111)

        if method == 'random':
            signs = self.random_signs(rows, self.neurons_per_layer)
            weights = np.multiply(
                np.random.rand(rows, self.neurons_per_layer), signs)
        elif method == 'xavier_he':
            stdv_in = np.sqrt(6 / self.number_of_inputs)
            stdv_rest = np.sqrt(6 / self.neurons_per_layer)

            # input weights
            signs = self.random_signs(self.number_of_inputs,
                                      self.neurons_per_layer)
            weights[1:self.number_of_inputs + 1] = np.multiply(
                np.random.rand(self.number_of_inputs, self.neurons_per_layer),
                signs) * stdv_in
            # weights of hidden layers
            start_hidden = self.number_of_inputs + 1
            for layer in range(self.hidden_layers - 1):
                signs = self.random_signs(self.neurons_per_layer,
                                          self.neurons_per_layer)
                start = start_hidden + (self.neurons_per_layer + 1) * layer + 1
                end = start + self.neurons_per_layer
                weights[start:end] = np.multiply(
                    np.random.rand(self.neurons_per_layer,
                                   self.neurons_per_layer), signs) * stdv_rest

            # output weights
            signs = self.random_signs(1, self.neurons_per_layer)
            weights[-1] = np.multiply(
                np.random.rand(1, self.neurons_per_layer), signs) * stdv_rest

        else:
            print("Could not find method {} for initialization."
                  .format(method))
            print("Possible options are 'random', 'xavier_he'")

        print("\nInitialized weights and biases for MLP with {} parameters.\n"
              .format(rows * self.neurons_per_layer))
        print("MLP structure:")
        print("--------------")
        print("Input features:    {}".format(self.number_of_inputs))
        print("Hidden layers:     {}".format(self.hidden_layers))
        print("Neurons per layer: {}".format(self.neurons_per_layer))
        print("Number of weights: {}".format(np.product(weights.shape)))
        return weights

    def network_output(self, x_in, weights):
        """Compute the forward pass through the MLP.

        Parameters
        ----------
        self - class instance : accesses weights, biases, and network structure
        x_in - array          : input vector

        Return
        ------
        N - scalar : feedforward network output

        """
        activations = self.sigmoid(
            np.dot(x_in, weights[1:self.number_of_inputs + 1]) + weights[0])
        # acivations for all hidden layers > 1
        for layer in range(0, self.hidden_layers - 1):
            start = (self.number_of_inputs + 1) + \
                    (self.neurons_per_layer + 1) * layer
            end = start + self.neurons_per_layer + 1
            activations = self.sigmoid(
                np.dot(activations, weights[start + 1:end]) + weights[start])
        # last layer: multiply with weights, add bias, and sum up
        return np.sum(np.multiply(weights[-1], activations) + weights[-2])

    @staticmethod
    def sigmoid(z):
        """Compute sigmoid function with array support."""
        return 1/(1+np.exp(-z))

    @staticmethod
    def random_signs(dim1, dim2):
        """Create and fill a 2D array randomly with either 1 or -1."""
        np.random.seed(1111)
        return np.asarray([
               np.random.choice([-1.0, 1.0])
               for i in range(dim1) for j in range(dim2)]).reshape(dim1, dim2)

    def update_weights(self, update):
        """Update the weights with some increment 'update'."""
        self.weights += update

    def set_weights(self, new_weights):
        """Set weights explicitly, for example after restart."""
        assert self.weights.shape == new_weights.shape
        self.weights = new_weights

    def get_weights(self):
        """Return the current weights."""
        return self.weights

    def write_weights_to_disk(self, folder):
        """Write current weights to disk."""
        file_path = "./" + folder + "/" + self.name + "/best_weights"
        if not os.path.exists("./" + folder + "/" + self.name + "/"):
            os.makedirs("./" + folder + "/" + self.name + "/")
        np.save(file_path, self.weights)
        print("Wrote weights to file {}".format(file_path))

    def read_weights_from_disk(self, file_path):
        """Read network weights from disk."""
        disk_weights = np.load(file_path)
        assert disk_weights.shape == self.weights.shape
        self.weights = disk_weights
        print("Loaded weights from file {}".format(file_path))

    def assert_input_length(self, list_to_test):
        """Assert input length of lists."""
        assert len(list_to_test) == self.number_of_inputs, \
            "Wrong number of arguments!"


class TrialFunctionDataBasedLearning(IBVProblem):
    """Class to learn based on data."""

    def __init__(self, name, mlp, training_data, beta, d_v):
        """Initialize a TrialFunctionDataBasedLearning object."""
        super(TrialFunctionDataBasedLearning, self).__init__(name, mlp)
        self.beta = beta
        self.d_v = d_v
        self.training = training_data

    def bc(self, x_in):
        """Compute part of trial function that fulfills boundary conditions.

        Parameters
        ----------
        x_in - array-like : input vector
                            [0] accesses the x-coordinate
                            [1] accesses the y-coordinate

        """
        return (1.0 - x_in[0]) \
            * (2.0 * self.mlp.sigmoid(self.beta * x_in[1]) - 1.0)

    def grad_network(self, x_in, weights):
        """Compute the gradient of N with respect to x_in."""
        def local_netwok_output(x_in, weights):
            return self.mlp.network_output(x_in, weights)
        return grad(local_netwok_output)(x_in, weights)

    def trial_function(self, x_in, weights):
        """Compute the trail function c_t."""
        net = self.mlp.network_output(x_in, weights)
        y_const = np.asarray([x_in[0], 1.0])
        net_y = self.mlp.network_output(y_const, weights)
        dy_net = self.grad_network(y_const, weights)[1]
        return self.bc(x_in) + x_in[0] * (1.0 - x_in[0]) * x_in[1] * \
            (net - net_y - dy_net)

    def reference_solution(self, ind):
        """Return reference solution at a given index."""
        return self.training[ind]

    def loss_function_kernel(self, weights, x_in, ind):
        """Return the data-based loss function."""
        ref = self.reference_solution(ind)
        trial = self.trial_function(x_in, weights)
        return (ref - trial)**2


class TrialFunctionResidualBasedLearning(IBVProblem):
    """Class to learn based on partial derivative residuals."""

    def __init__(self, name, mlp, beta, d_v):
        """Initialize a TrialFunctionResidualBasedLearning object."""
        super(TrialFunctionResidualBasedLearning, self).__init__(name, mlp)
        self.beta = beta
        self.d_v = d_v

    def bc(self, x_in):
        """Compute part of trial function that fulfills boundary conditions.

        Parameters
        ----------
        x_in - array-like : input vector
                            [0] accesses the x-coordinate
                            [1] accesses the y-coordinate

        """
        return (1.0 - x_in[0]) \
            * (2.0 * self.mlp.sigmoid(self.beta * x_in[1]) - 1.0)

    def dy_bc(self, x_in):
        """Compute y-derivative of bc function."""
        sig = self.mlp.sigmoid(self.beta * x_in[1])
        return 2.0 * self.beta * (sig - sig**2)

    def dyy_bc(self, x_in):
        """Compute yy-derivative of bc function."""
        sig = self.mlp.sigmoid(self.beta * x_in[1])
        return 2.0 * self.beta**2 * (2.0 * sig**3 - 3.0 * sig**2 + sig)

    def grad_network(self, x_in, weights):
        """Compute network gradient with respect to x_in."""
        def local_netwok_output(x_in, weights):
            return self.mlp.network_output(x_in, weights)
        return grad(local_netwok_output)(x_in, weights)

    def laplace_network(self, x_in, weights):
        """Compute network laplace with respect to x_in."""
        def local_netwok_output(x_in, weights):
            return self.mlp.network_output(x_in, weights)
        return egrad(egrad(local_netwok_output))(x_in, weights)

    def dy_grad_network(self, x_in, weights):
        """Compute the y-derivative of the network gradient."""
        def dx_network(x_in, weights):
            return self.grad_network(x_in, weights)[0]
        return grad(dx_network)(x_in, weights)[1]

    def dy_laplace_network(self, x_in, weights):
        """Compute the y-derivative of the network's Laplace."""
        def dxx_network(x_in, weights):
            return self.laplace_network(x_in, weights)[0]
        return grad(dxx_network)(x_in, weights)[1]

    def trial_function(self, x_in, weights):
        """Compute the trail function c_t."""
        net = self.mlp.network_output(x_in, weights)
        y_const = np.asarray([x_in[0], 1.0])
        net_y = self.mlp.network_output(y_const, weights)
        dy_net = self.grad_network(y_const, weights)[1]
        return self.bc(x_in) + x_in[0] * (1.0 - x_in[0]) * x_in[1] * \
            (net - net_y - dy_net)

    def laplace_trial_function(self, x_in, weights):
        """Compute the Laplace of the trial function."""
        xy1 = np.asarray([x_in[0], 1.0])
        net = self.mlp.network_output(x_in, weights)    # network at x, y
        net_y1 = self.mlp.network_output(xy1, weights)  # network at x, 1
        grad_net = self.grad_network(x_in, weights)
        grad_net_y1 = self.grad_network(xy1, weights)
        dx_net = grad_net[0]                            # dx network at x, y
        dy_net = grad_net[1]                            # dy network at x, y
        dx_net_y1 = grad_net_y1[0]                      # dx network at x, 1
        dy_net_y1 = grad_net_y1[1]                      # dy network at x, 1
        laplace_net = self.laplace_network(x_in, weights)
        laplace_net_y1 = self.laplace_network(xy1, weights)
        dxx_net = laplace_net[0]                        # dxx network at x, y
        dyy_net = laplace_net[1]                        # dyy network at x, y
        dxx_net_y1 = laplace_net_y1[0]                  # dxx network at x, 1
        dyx_net_y1 = self.dy_grad_network(xy1, weights)  # dyx network at x, 1
        dyxx_net_y1 = self.dy_laplace_network(xy1, weights)  # dyxx net at x, 1
        dxx = 2 * x_in[1] * (
            0.5 * x_in[0]
            * (1.0 - x_in[0]) * (dxx_net - dxx_net_y1 - dyxx_net_y1)
            + (1.0 - 2.0 * x_in[0]) * (dx_net - dx_net_y1 - dyx_net_y1)
            - (net - net_y1 - dy_net_y1)
        )
        dyy = (1.0 - x_in[0]) * (
            self.dyy_bc(x_in)
            + 2.0 * x_in[0] * dy_net
            + x_in[0] * x_in[1] * dyy_net
        )
        return np.asarray([dxx, dyy])

    def dy_trial_function(self, x_in, weights):
        """Compute the Laplace of the trail function."""
        net = self.mlp.network_output(x_in, weights)
        dy_net = self.grad_network(x_in, weights)[1]
        xy1 = np.asarray([x_in[0], 1.0])
        net_y1 = self.mlp.network_output(xy1, weights)
        dy_net_y1 = self.grad_network(xy1, weights)[1]
        dy = (1.0 - x_in[0]) \
            * (self.dy_bc(x_in) + x_in[0] * (net - net_y1 - dy_net_y1)
                + x_in[0] * x_in[1] * dy_net)
        return dy

    def loss_function_kernel(self, weights, x_in, ind):
        """Compute residual based loss function kernel."""
        dy_trial = self.dy_trial_function(x_in, weights)
        dxx_dyy_trial = self.laplace_trial_function(x_in, weights)
        return (dy_trial - self.d_v * sum(dxx_dyy_trial))**2


class PureNetworkLearning(IBVProblem):
    """Class to train the pure network based on data."""

    def __init__(self, name, mlp, training_data, beta, d_v, penalty=1.0):
        """Initialize a PureNetworkLearning object."""
        super(PureNetworkLearning, self).__init__(name, mlp)
        self.beta = beta
        self.d_v = d_v
        self.training = training_data
        self.penalty = penalty

    def trial_function(self, x_in, weights):
        """Return the network output (required by super class)."""
        return self.mlp.network_output(x_in, weights)

    def reference_solution(self, ind):
        """Return reference solution at a given index."""
        return self.training[ind]

    def loss_function_kernel(self, weights, x_in, ind):
        """Compute loss function with or without penalty factor."""
        net = self.mlp.network_output(x_in, weights)
        ref = self.reference_solution(ind)
        p_factor = 1.0
        is_boundary = (np.product(x_in) < 1.0E-12) or \
                      (abs(x_in[0] - 1.0) < 1.0E-12) or \
                      (abs(x_in[1] - 1.0) < 1.0E-12)
        if is_boundary:
            p_factor = self.penalty
        return ((ref - net) * p_factor)**2


class CustomFunctionLearning(IBVProblem):
    """Class to learn based on data."""

    def __init__(self, name, mlp, training_data, beta, d_v):
        """Initialize a CustomFunctionLearning object."""
        super(CustomFunctionLearning, self).__init__(name, mlp)
        self.beta = beta
        self.d_v = d_v
        self.training = training_data

    def bc(self, x_in):
        """Compute part of trial function that fulfills boundary conditions.

        Parameters
        ----------
        x_in - array-like : input vector
                            [0] accesses the x-coordinate
                            [1] accesses the y-coordinate

        """
        return (2.0 * self.mlp.sigmoid(self.beta * x_in[1]) - 1.0)

    def trial_function(self, x_in, weights):
        """Compute custom trial function."""
        return self.bc(x_in) * \
            (1.0 - erf(x_in[0] / (np.sqrt(x_in[1]*4.0*self.d_v) + 1.0E-9) *
                       np.absolute(self.mlp.network_output(x_in, weights))))

    def reference_solution(self, ind):
        """Return reference solution at a given index."""
        return self.training[ind]

    def loss_function_kernel(self, weights, x_in, ind):
        """Compute loss function based on data."""
        net = self.trial_function(x_in, weights)
        ref = self.reference_solution(ind)
        return (ref - net)**2


def erf(x):
    """Compute the error function."""
    # constants
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    # A&S formula 7.1.26
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp(-x*x)
    return y
