import numpy as np
from nova.nn import Layer, Activation, NN, Operator


def backprop(layers: list[Layer], loss: Operator):
    layers[-1].eval_grads(loss)
    for layer in reversed(layers[:-1]):
        layer.eval_grads()


class TestLayer:

    def test_gradient_wrt_weights(self):
        input_size, output_size = 3, 1

        output_layer = Layer(output_size)
        input_layer = Layer(input_size, next_layer=output_layer)

        input_layer.add_arch(
            weights=np.ones(input_size),
            bias=np.zeros(output_size),
            activation=Activation.identity,
        )

        layers = [input_layer, output_layer]

        for _ in range(5):
            input_values = np.random.rand(input_size)
            input_layer.fill(input_values)
            input_layer.propagate()

            backprop(layers=layers, loss=Activation.identity)
            assert (input_values == input_layer.grad_w).all()

            output_layer.clear(), input_layer.clear()

    def test_gradient_wrt_values(self):
        input_size, output_size = 3, 2

        output_layer = Layer(output_size)
        input_layer = Layer(input_size, next_layer=output_layer)

        easy_weights = np.array([[1, 2, 3], [4, 5, 6]])
        input_layer.add_arch(
            weights=easy_weights,
            bias=np.zeros(output_size),
            activation=Activation.identity,
        )

        layers = [input_layer, output_layer]

        input_values = np.ones(input_size)
        input_layer.fill(input_values)
        input_layer.propagate()

        backprop(layers=layers, loss=Activation.identity)
        assert (input_layer.grad_snapshots[0] == np.array([5, 7, 9])).all()


class TestNN:

    def test_arch(self):
        arch = np.array([3, 4, 2])
        nn = NN(arch=arch)

        for layer, desired_size in zip(nn.layers, arch):
            assert layer.size == desired_size

    def test_forward_pass(self):
        arch = np.array([1, 1, 1])
        activations = [Activation.identity, Activation.identity]
        nn = NN(arch=arch, activations=activations)

        nn.forward(np.array([1.0]))

        for in_layer, out_layer in zip(nn.hidden_layers, nn.conjugate_layers):
            a, x, b = in_layer.weights, in_layer.values, in_layer.bias
            y = out_layer.values
            assert y == a * x + b
