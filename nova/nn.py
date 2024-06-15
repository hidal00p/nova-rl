from typing import Optional, Callable, TypeAlias
import numpy as np

"""
My goal is to store gradients per pass, so that I could
perform mini batch average updates.
"""


class NameSpace:
    """
    This class is used strictly as a namespace
    """

    def __new__(cls, *args, **kwargs):
        raise RuntimeError(
            f"{cls.__name__} is not meant for instantiation. It is used strictly as a namespace."
        )


class Activation(NameSpace):
    @classmethod
    def tanh(cls, x: np.ndarray, der: bool = False) -> np.ndarray:
        return 1 - np.power(np.tanh(x), 2) if der else np.tanh(x)

    @classmethod
    def relu(cls, x: np.ndarray, der: bool = False) -> np.ndarray:
        return 1 * (x > 0) if der else x * (x > 0)

    @classmethod
    def softmax(cls, x: np.ndarray, der: bool = False) -> np.ndarray:
        raise NotImplementedError()

    @classmethod
    def identity(cls, x: np.ndarray, der=False) -> np.ndarray:
        return np.ones(x.shape) if der else x


class Loss(NameSpace):

    @classmethod
    def single_return(cls, log_prob: float, der: bool = False):
        pass


Operator: TypeAlias = Callable[..., np.ndarray]
GradSnapshot: TypeAlias = tuple[np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]


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
        self.grad_snapshots: list[GradSnapshot] = []

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
        self.grad_b = np.zeros(self.bias.shape)

    @only_hidden_layer
    def propagate(self):
        # Compute forward pass for the next layer
        linear_result = np.matmul(self.weights, self.values) + self.bias
        activation_result = self.activation(linear_result)

        # Fill the next layer with the computation
        self.next_layer.fill(activation_result)

        # Compute and store all the local gradients
        grad_activation_result = self.activation(linear_result, der=True)
        self.grad_values[:] = np.matmul(self.weights.T, grad_activation_result)[:]
        self.grad_w[:] = np.outer(grad_activation_result, self.values)[:]
        self.grad_b[:] = grad_activation_result[:]

        # Store gradients of the current propagation routine
        self.grad_snapshots.append(
            (self.grad_values[:], self.grad_w[:], self.grad_b[:])
        )

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

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass of the constructed neural network.
        """
        self.input_layer.fill(x)

        for layer in self.hidden_layers:
            layer.propagate()

        return self.output_layer.values

    def backward(self):
        """
        Perform a backward propagtion of gradients.
        """
        pass
