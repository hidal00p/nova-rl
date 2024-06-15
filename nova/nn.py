from typing import Optional, Callable, TypeAlias
import numpy as np

"""
My goal is to store gradients per pass, so that I could
perform mini batch average updates.
"""


class Activation:
    """
    This class is used strictly as a namespace
    """

    def __new__(*args, **kwargs):
        raise RuntimeError("Activations is not meant for instantiation")

    @classmethod
    def tanh(cls, x: np.ndarray, der: bool = False) -> np.ndarray:
        if der:
            return np.tanh(x)
        else:
            return 1 - np.power(np.tanh(x), 2)

    @classmethod
    def relu(cls, x: np.ndarray, der: bool = False) -> np.ndarray:
        if der:
            return np.array([])
        else:
            return np.array([])

    @classmethod
    def softmax(cls, x: np.ndarray, der: bool = False) -> np.ndarray:
        if der:
            return np.array([])
        else:
            return np.array([])

    @classmethod
    def identity(cls, x: np.ndarray, der=False) -> np.ndarray:
        return np.ones(x.shape) if der else x


Operator: TypeAlias = Callable[..., np.ndarray]


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
        self.grad_values = np.zeros(self.size)

    def connect_layer(self, next_layer: "Layer"):
        self.next_layer = next_layer

    @only_hidden_layer
    def add_arch(self, weights: np.ndarray, bias: np.ndarray, activation: Operator):
        self.weights = (
            weights if len(weights.shape) > 1 else weights.reshape(1, weights.shape[0])
        )
        self.bias = bias
        self.activation = activation

        self.grad_w = np.zeros(self.weights.shape)
        self.grad_b = np.zeros(self.size)

    @only_hidden_layer
    def propagate(self):
        if self.next_layer is None:
            return

        # Compute forward pass for the next layer
        linear_result = np.matmul(self.weights, self.values) + self.bias
        activation_result = self.activation(linear_result)

        # Fill the next layer with the computation
        self.next_layer.fill(activation_result)

        # Compute and store all the local gradients
        grad_activation_result = self.activation(linear_result, der=True)
        self.grad_values = np.matmul(self.weights.T, grad_activation_result)
        self.grad_w = np.outer(grad_activation_result, self.values)

    def fill(self, values: np.ndarray):
        # Just copy the values into allocated array
        self.values[:] = values[:]

    def clear(self):
        self.values[:] = 0.0
        self.grad_values[:] = 0.0

        if self.next_layer is not None:
            self.grad_w[:] = 0.0
            self.grad_b[:] = 0.0


class NN:
    def __init__(
        self,
        arch: np.ndarray,
    ):
        self.arch = arch
        self.layers: list[Layer] = [Layer(size) for size in arch]

        self.construct_layers()

    def construct_layers(self):
        for in_layer, out_layer in zip(self.layers[:-1], self.layers[1:]):
            input_dim, output_dim = in_layer.size, out_layer.size
            weights = np.random.rand(output_dim, input_dim)
            bias = np.random.random(output_dim)
            in_layer.connect_layer(out_layer)
            in_layer.add_arch(weights=weights, bias=bias, activation=Activation.tanh)