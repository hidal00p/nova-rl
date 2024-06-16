from typing import Optional
import numpy as np

from nova.utils import Operator, Activation

"""
My goal is to store gradients per pass, so that I could
perform mini batch average updates.
"""


def only_hidden_layer(func):
    def _func(self: "Layer", *args, **kwargs):
        assert (
            self.next_layer is not None
        ), f"Function {func.__name__} is meant only for the hidden layer."

        return func(self, *args, **kwargs)

    return _func


class Layer:
    """
    Layer represents a layer of neurons in a neural network.
    It forms a neural network via a linked list structure,
    where each layer stores a pointer to the next layer, unless
    it is the final layer.
    """

    def __init__(
        self,
        size: int,
        next_layer: Optional["Layer"] = None,
    ):
        self.size = size
        self.next_layer = next_layer
        self.values = np.zeros(self.size)

        self.value_snapshots: list[np.ndarray] = []
        self.grad_snapshots: list[np.ndarray] = []

    def connect_layer(self, next_layer: "Layer"):
        self.next_layer = next_layer

    @only_hidden_layer
    def add_arch(self, weights: np.ndarray, bias: np.ndarray, activation: Operator):
        self.weights = (
            weights if len(weights.shape) > 1 else weights.reshape(1, weights.shape[0])
        )
        self.bias = bias
        self.activation = activation
        self.activation_grad_snapshots: list[np.ndarray] = []

        self.grad_w = np.zeros(self.weights.shape)
        self.grad_b = np.zeros(self.bias.shape)

    @only_hidden_layer
    def propagate(self):
        # Compute forward pass for the next layer
        linear_result = np.matmul(self.weights, self.values) + self.bias

        # Fill the next layer with the computation
        self.next_layer.fill(self.activation(linear_result))

        # Compute and store all the local gradients
        activation_grad = self.activation(linear_result, der=True)
        self.activation_grad_snapshots.append(activation_grad.copy())

    def fill(self, values: np.ndarray):
        # Just copy the values into allocated array
        self.values[:] = values[:]
        self.value_snapshots.append(self.values.copy())

    def clear(self):
        self.values[:] = 0.0
        self.value_snapshots.clear()
        self.grad_snapshots.clear()

        if self.next_layer is not None:
            self.grad_w[:] = 0.0
            self.grad_b[:] = 0.0
            self.activation_grad_snapshots.clear()

    def eval_grads(self, loss: Optional[Operator] = None):
        """
        For correct evaluation gradients have to be backward evaluated.
        Hence, the gradients w.r.t to values must be available to the
        current layer.
        """
        if self.next_layer is None:
            assert loss is not None
            # TODO: Can I vectorize this?
            # Maybe by passing a full array into the loss function at ones
            # it will be able to correctly compute losses per snapshot.
            for values in self.value_snapshots:
                self.grad_snapshots.append(loss(values, der=True))

        else:
            for values, activation_grad, loss_grad in zip(
                self.value_snapshots,
                self.activation_grad_snapshots,
                self.next_layer.grad_snapshots,
            ):
                activation_grad = np.multiply(activation_grad, loss_grad)

                self.grad_snapshots.append(np.matmul(self.weights.T, activation_grad))
                self.grad_w += np.outer(activation_grad, self.values)
                self.grad_b += activation_grad

            self.grad_w /= len(self.value_snapshots)
            self.grad_b /= len(self.value_snapshots)


class NN:
    def __init__(self, arch: np.ndarray, activations: Optional[list[Operator]] = None):
        assert len(arch) > 1

        self.arch = arch
        self.layers: list[Layer] = [Layer(size) for size in arch]
        self.activations = activations or [Activation.tanh] * (len(self.layers) - 1)

        self.__construct_layers()

    @property
    def hidden_layers(self):
        return self.layers[:-1]

    @property
    def conjugate_layers(self):
        """
        Layers conjugate to hidden layers.
        """
        return self.layers[1:]

    @property
    def input_layer(self):
        return self.layers[0]

    @property
    def output_layer(self):
        return self.layers[-1]

    def __construct_layers(self):
        """
        Connects layers together, and injects weights, biases and activations into them.
        """
        for in_layer, out_layer, activation in zip(
            self.hidden_layers, self.conjugate_layers, self.activations
        ):
            input_dim, output_dim = in_layer.size, out_layer.size
            weights = np.random.rand(output_dim, input_dim)
            bias = np.random.random(output_dim)
            in_layer.connect_layer(out_layer)
            in_layer.add_arch(weights=weights, bias=bias, activation=activation)

    def clear(self):
        for layer in self.layers:
            layer.clear()

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass of the constructed neural network.
        """
        self.input_layer.fill(x)

        for layer in self.hidden_layers:
            layer.propagate()

        return self.output_layer.values

    def backward(self, loss: Operator):
        """
        Perform a backward propagtion of gradients.
        """
        self.output_layer.eval_grads(loss)
        for layer in reversed(self.hidden_layers):
            layer.eval_grads()

    @property
    def params(self):
        """
        Let's construct a vector out of all neural net params.

        TODO: this is not efficient, I would like to get rid of arrays completely.
        """
        return [layer.weights.ravel() for layer in self.hidden_layers] + [
            layer.bias for layer in self.layers
        ]

    @property
    def grads(self):
        """
        Let's construct a vector out of all grads, which were computed during the backward call.

        TODO: this is not efficient, I would like to get rid of arrays completely.
        """
        return [layer.grad_w.ravel() for layer in self.hidden_layers] + [
            layer.grad_b for layer in self.layers
        ]
