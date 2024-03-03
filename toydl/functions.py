import numpy as np
from toydl.core import Function, Variable, as_variable
from toydl import utils


class Tanh(Function):
    @Function.tuplize
    def forward(self, x, /):
        return np.tanh(x)

    @Function.tuplize
    def backward(self, gy, /):
        # y是弱引用
        y = self.outputs[0]()
        return gy * (1 - y * y)


class Sin(Function):
    @Function.tuplize
    def forward(self, x, /):
        return np.sin(x)

    @Function.tuplize
    def backward(self, gy, /):
        x = self.inputs[0]
        return cos(x) * gy


class Cos(Function):
    @Function.tuplize
    def forward(self, x, /):
        return np.cos(x)

    @Function.tuplize
    def backward(self, gy, /):
        x = self.inputs[0]
        return -sin(x) * gy


class Reshape(Function):
    def __init__(self, y_shape):
        self.x_shape = None
        self.y_shape = y_shape

    @Function.tuplize
    def forward(self, x, /):
        self.x_shape = x.shape
        return x.reshape(self.y_shape)

    @Function.tuplize
    def backward(self, gy, /):
        return gy.reshape(self.x_shape)


class Transpose(Function):
    @Function.tuplize
    def forward(self, x, /):
        return np.transpose(x)

    @Function.tuplize
    def backward(self, gy, /):
        return np.transpose(gy)


class Sum(Function):
    def __init__(self, axis, keepdims, x_shape):
        self.axis = axis
        self.keepdims = keepdims
        self.x_shape = x_shape

    @Function.tuplize
    def forward(self, x, /):
        return x.sum(axis=self.axis, keepdims=self.keepdims)

    @Function.tuplize
    def backward(self, gy, /):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        return broadcast_to(gy, self.x_shape)


class BroadcastTo(Function):
    def __init__(self, x_shape, y_shape):
        self.x_shape = x_shape
        self.y_shape = y_shape

    @Function.tuplize
    def forward(self, x, /):
        return np.broadcast_to(x, self.y_shape)

    @Function.tuplize
    def backward(self, gy, /):
        return sum_to(gy, self.x_shape)


class SumTo(Function):
    def __init__(self, x_shape, y_shape):
        self.x_shape = x_shape
        self.y_shape = y_shape

    @Function.tuplize
    def forward(self, x, /):
        return utils.sum_to(x, self.y_shape)

    @Function.tuplize
    def backward(self, gy, /):
        return broadcast_to(gy, self.x_shape)


class Matmul(Function):
    @Function.tuplize
    def forward(self, A, B, /):
        return A.dot(B)

    @Function.tuplize
    def backward(self, gy, /):
        A, B = self.inputs
        return matmul(gy, B.T), matmul(A.T, gy)


class MeanSquaredError(Function):
    @Function.tuplize
    def forward(self, x0, x1, /):
        diff = x0 - x1
        return np.sum(diff ** 2) / len(diff)

    @Function.tuplize
    def backward(self, gy, /):
        x0, x1 = self.inputs
        n = len(x0)
        gx0 = gy * 2 * (x0 - x1) / n
        gx1 = -gx0
        return gx0, gx1


class Linear(Function):
    @Function.tuplize
    def forward(self, x, W, b, /):
        return np.dot(x, W) + b

    @Function.tuplize
    def backward(self, gy, /):
        x, W, b = self.inputs
        # 考虑可能广播, gy和b的形状不一定一样, 需要使用广播的反向传播逻辑
        gx, gW, gb = matmul(gy, W.T), matmul(x.T, gy), sum_to(gy, b.shape)
        return gx, gW, gb


class Sigmoid(Function):
    @Function.tuplize
    def forward(self, x, /):
        return np.tanh(x * 0.5) * 0.5 + 0.5

    @Function.tuplize
    def backward(self, gy, /):
        y = self.outputs[0]()
        return gy * y * (1 - y)


def tanh(x, /) -> list[Variable] | Variable:
    return Tanh()(x)


def sin(x, /) -> list[Variable] | Variable:
    return Sin()(x)


def cos(x, /) -> list[Variable] | Variable:
    return Cos()(x)


def reshape(x, /, y_shape) -> list[Variable] | Variable:
    # new_shape是函数的参数而不是数据的一部分
    if x.shape == y_shape:
        return as_variable(x)
    return Reshape(y_shape)(x)


def transpose(x, /) -> list[Variable] | Variable:
    return Transpose()(x)


def sum(x, /, axis=None, keepdims=False) -> list[Variable] | Variable:
    return Sum(axis, keepdims, x.shape)(x)


def broadcast_to(x, /, shape) -> list[Variable] | Variable:
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(x.shape, shape)(x)


def sum_to(x, /, shape) -> list[Variable] | Variable:
    if x.shape == shape:
        return as_variable(x)
    return SumTo(x.shape, shape)(x)


def matmul(A, B, /) -> list[Variable] | Variable:
    return Matmul()(A, B)


def mean_squared_error(x0, x1, /) -> Variable:
    return MeanSquaredError()(x0, x1)


def mean_squared_error_simple(x0, x1, /) -> Variable:
    diff = x0 - x1
    return sum(diff ** 2) / len(diff)


def linear_simple(x, W, b=None):
    t = matmul(x, W)
    if b is None:
        return t

    y = t + b
    t.data = None
    return y


def linear(x, W, b=.0):
    return Linear()(x, W, b)


def sigmoid_simple(x, /):
    return 1 / (1 + exp(-as_variable(x)))


def sigmoid(x, /):
    return Sigmoid()(x)