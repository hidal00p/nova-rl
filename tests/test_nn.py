import numpy as np
from nova.nn import Layer, Activation, NN


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

        for _ in range(5):
            input_values = np.random.rand(input_size)
            input_layer.fill(input_values)
            input_layer.propagate()
            assert (input_values == input_layer.grad_w).all()

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

        input_values = np.ones(input_size)
        input_layer.fill(input_values)
        input_layer.propagate()

        assert (input_layer.grad_values == np.array([5, 7, 9])).all()


class TestNN:

    def test_arch(self):
        arch = np.array([3, 4, 2])
        nn = NN(arch=arch)

        for layer, desired_size in zip(nn.layers, arch):
            assert layer.size == desired_size
