from typing import Callable, TypeAlias
import numpy as np

Operator: TypeAlias = Callable[..., np.ndarray]


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
