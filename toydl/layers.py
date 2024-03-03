import numpy as np
from toydl import Parameter
import toydl.functions as F
import weakref


class Layer:
    def __init__(self):
        self._params = set()

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs, )
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs):
        raise NotImplementedError

    # 一个生成所有输入的生成器
    def params(self):
        for name in self._params:
            yield self.__dict__[name]

    def clear_grads(self):
        for param in self.params():
            param.clear_grad()


class Linear(Layer):
    def __init__(self, out_size, in_size=None, nobias=False, dtype=np.float32):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype
        self.b = None if nobias else Parameter(np.zeros(out_size, dtype=dtype), name='b')
        # 推迟创建W.data的时间, 通过用户输入自动推理输入大小
        self.W = Parameter(None, name='W')
        if self.in_size is not None:
            self._init_W()

    def _init_W(self):
        I, O = self.in_size, self.out_size
        self.W.data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)

    def forward(self, x):
        if self.W.data is None:
            # 输入的x一行是一个数据点, 故第一维是样本数量, 第二维才是数据点维数
            self.in_size = x.shape[1]
            self._init_W()

        y = F.linear(x, self.W, self.b)
        return y

