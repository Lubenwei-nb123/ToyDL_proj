import numpy as np
from typing import Iterable, Type
from heapq import heappush, heappop
import weakref
import contextlib


class Config:
    enable_backprop = True


class Variable:
    # 设置更高的运算符优先级使得自己的正向内置方法被优先调用
    __array_priority__ = 2200

    def __init__(self, data, name=None):
        if not isinstance(data, np.ndarray) and data is not None:
            data = np.array(np.float64(data))
        self.name = name
        self._data = data
        self._grad = None
        self._creator = None
        self._generation = 0

    @property
    def data(self):
        return self._data

    @property
    def grad(self):
        return self._grad

    @property
    def creator(self):
        return self._creator

    @property
    def generation(self):
        return self._generation

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self) -> int:
        return len(self.data)

    def __neg__(self):
        return neg(self)

    def __repr__(self) -> str:
        prefix = f'{self.__class__.__name__}('
        postfix = ')'
        return prefix + str(self.data).replace('\n', ', ') + postfix

    def __str__(self) -> str:
        prefix = f'{self.__class__.__name__}('
        postfix = ')'
        return prefix + str(self.data).replace('\n', '\n' + ' ' * len(prefix)) + postfix

    def __iter__(self) -> Iterable:
        return (e for e in self.data)

    def __add__(self, other):
        try:
            return add(self, other)
        except TypeError:
            return NotImplementedError

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        try:
            # 用sub模块来代替neg和add模块, 更精简
            return sub(self, other)
        except TypeError:
            return NotImplementedError

    def __rsub__(self, other):
        try:
            # 用sub模块来代替neg和add模块, 更精简
            return sub(other, self)
        except TypeError:
            return NotImplementedError

    def __mul__(self, other):
        try:
            return mul(self, other)
        except TypeError:
            return NotImplementedError

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        try:
            return div(self, other)
        except TypeError:
            return NotImplementedError

    def __rtruediv__(self, other):
        try:
            return div(other, self)
        except TypeError:
            return NotImplementedError

    def __pow__(self, other):
        try:
            return pow(self, other)
        except TypeError:
            return NotImplementedError

    def __rpow__(self, other):
        try:
            return pow(other, self)
        except TypeError:
            return NotImplementedError

    def __eq__(self, other):
        return self.data == other

    def set_creator(self, func):
        self._creator = func
        self._generation = func.generation + 1

    def backward(self, retain_grad=False):
        """改成循环的好处是可以在函数内初始化梯度"""
        if self.grad is None:
            self._grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f: Function):
            if f not in seen_set:
                '''先对generation取反'''
                f._generation = -f._generation
                '''使用小根堆算法以O(1)时间取出最小值(即最大的generation)'''
                heappush(funcs, f)
                seen_set.add(f)

        add_func(self._creator)
        '''使用广度优先遍历的变种来反向传播计算图'''
        while funcs:
            f = heappop(funcs)
            f._generation = -f._generation
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x._grad = gx
                else:
                    '''不能用+=是因为有的函数子类gx和gy是绑定在一起的, 如果使用原地操作会导致gy也被修改'''
                    x._grad = x._grad + gx
                if x.creator is not None:
                    add_func(x._creator)

            if not retain_grad:
                for output in f._outputs:
                    output()._grad = None

    def clear_grad(self):
        self._grad = None


class Function:
    def __call__(self, *inputs: Variable) -> list[Variable] | Variable:
        inputs = [as_variable(x) for x in inputs]

        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        outputs = [Variable(y) for y in ys]

        if Config.enable_backprop:
            self._generation = max([x.generation for x in inputs])
            '''仅在允许反向传播时创建计算图的连接, 禁用时会通过引用计数机制回收中间结果'''
            for output in outputs:
                output.set_creator(self)

        self._inputs = inputs
        '''使用弱引用避免循环引用造成占用内存过大'''
        self._outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(x)'

    def __lt__(self, other) -> bool | Type[NotImplementedError]:
        try:
            return self.generation < other.generation
        except (TypeError, AttributeError):
            return NotImplementedError

    def forward(self, in_data):
        raise NotImplementedError

    def backward(self, gy):
        raise NotImplementedError

    @staticmethod
    def tuplize(func):
        """将输出元组化"""
        def tuple_res(*xs):
            ys = func(*xs)
            if not isinstance(ys, tuple):
                ys = (ys,)
            return ys
        return tuple_res

    @property
    def inputs(self):
        """使得属性在外部只读"""
        return self._inputs

    @property
    def outputs(self):
        """使得属性在外部只读"""
        return self._outputs

    @property
    def generation(self):
        return self._generation


class Square(Function):
    @Function.tuplize
    def forward(self, x, /):
        return x ** 2

    @Function.tuplize
    def backward(self, gy, /):
        return 2 * self._inputs[0].data * gy


class Exp(Function):
    @Function.tuplize
    def forward(self, x, /):
        return np.exp(x)

    @Function.tuplize
    def backward(self, gy, /):
        return np.exp(self._inputs[0].data) * gy


class Add(Function):
    @Function.tuplize
    def forward(self, x0, x1, /):
        return x0 + x1

    @Function.tuplize
    def backward(self, gy, /):
        return gy, gy


class Mul(Function):
    @Function.tuplize
    def forward(self, x0, x1, /):
        return x0 * x1

    @Function.tuplize
    def backward(self, gy, /):
        return self._inputs[1].data * gy, self._inputs[0].data * gy


class Neg(Function):
    @Function.tuplize
    def forward(self, x, /):
        return -x

    @Function.tuplize
    def backward(self, gy, /):
        return -gy


class Sub(Function):
    @Function.tuplize
    def forward(self, x0, x1, /):
        return x0 - x1

    @Function.tuplize
    def backward(self, gy):
        return gy, -gy


class Div(Function):
    @Function.tuplize
    def forward(self, x0, x1):
        return x0 / x1

    @Function.tuplize
    def backward(self, gy):
        x0, x1 = self._inputs[0].data, self._inputs[1].data
        return gy / x1, gy * (-x0 / x1 ** 2)


class Pow(Function):
    @Function.tuplize
    def forward(self, x, c, /):
        return x ** c

    @Function.tuplize
    def backward(self, gy):
        x, c = self._inputs[0].data, self._inputs[1].data
        return gy * c * (x ** (c - 1)), gy * (x ** c) * np.log(x)


class Sin(Function):
    @Function.tuplize
    def forward(self, x):
        return np.sin(x)

    @Function.tuplize
    def backward(self, gy):
        x = self.inputs[0].data
        return np.cos(x) * gy


class Cos(Function):
    @Function.tuplize
    def forward(self, x, /):
        return np.cos(x)

    @Function.tuplize
    def backward(self, gy):
        x = self.inputs[0].data
        return -np.sin(x) * gy


@contextlib.contextmanager
def using_config(cls, attr, new_config=True):
    """创建上下文管理器来控制模式"""
    old_config = getattr(cls, attr)
    setattr(cls, attr, new_config)
    try:
        '''若用户代码出现异常, 会上浮到该管理器yield语句位置, 故需要放在try块内'''
        yield
    finally:
        setattr(cls, attr, old_config)


def no_grad(cls):
    return cls.using_config('enable_backprop', False)


def as_array(x) -> np.ndarray:
    if np.isscalar(x):
        return np.array(x)
    return x


def as_variable(obj) -> Variable:
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


def add(x0, x1, /) -> list[Variable] | Variable:
    return Add()(x0, x1)


def mul(x0, x1, /) -> list[Variable] | Variable:
    return Mul()(x0, x1)


def neg(x, /) -> list[Variable] | Variable:
    return Neg()(x)


def sub(x0, x1, /) -> list[Variable] | Variable:
    return Sub()(x0, x1)


def div(x0, x1, /) -> list[Variable] | Variable:
    return Div()(x0, x1)


def pow(x, c, /) -> list[Variable] | Variable:
    return Pow()(x, c)


def sin(x, /) -> list[Variable] | Variable:
    return Sin()(x)