from toydl import Variable, Function
import numpy as np
import os
import subprocess


def _var_code(v: Variable, verbose=False) -> str:
    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += str(v.shape) + ' ' if v.ndim != 0 else ''
        name += str(v.dtype)
    color = "gray" if v.name is None else 'orange'
    var_code = f'{id(v)} [label="{name}", color={color}, style=filled]\n'
    return var_code


def _func_code(f: Function) -> str:
    func_code = f'{id(f)} [label="{f.__class__.__name__}", color=lightblue, style=filled, shape=box]\n'
    for x in f.inputs:
        func_code += f'{id(x)} -> {id(f)}\n'
    for y in f.outputs:
        func_code += f'{id(f)} -> {id(y())}\n'
    return func_code


def get_dot_graph(y, verbose=False) -> str:
    """可视化计算图对应代码"""

    # 生成节点和连接的代码
    code = ''

    # 下面使用和反向传播一样的思路遍历整个计算图
    funcs = []
    seen_set = set()

    def add_func(f: Function):
        if f not in seen_set:
            # 这里不需要严格按照辈分顺序
            funcs.append(f)
            seen_set.add(f)
    add_func(y.creator)
    # 初始化代码
    code += _var_code(y, verbose)
    '''使用广度优先遍历的变种来反向传播计算图'''
    while funcs:
        f = funcs.pop()
        code += _func_code(f)
        for x in f.inputs:
            code += _var_code(x, verbose)
            if x.creator is not None:
                add_func(x.creator)

    return 'digraph g {\n' + code + "}"


def plot_dot_graph(output, verbose=False, to_file='graph.png'):
    dot_graph = get_dot_graph(output, verbose)

    # 将图片和dot文件保存在当前目录的子目录下
    tmp_dir = '.graph'
    # print(tmp_dir)
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    graph_path = os.path.join(tmp_dir, os.path.splitext(to_file)[0] + '.dot')
    to_file = os.path.join(tmp_dir, to_file)

    with open(graph_path, 'w') as f:
        f.write(dot_graph)

    extension = os.path.splitext(to_file)[1][1:]
    cmd = 'dot {} -T {} -o {}'.format(graph_path, extension, to_file)
    show = 'start {}'.format(to_file)
    # print(cmd)
    subprocess.run(cmd, shell=True)
    # 自动
    subprocess.run(show, shell=True)


def reshape_sum_backward(gy, x_shape, axis, keepdims):
    # 根据axis和keepdims参数调整gy真正的形状
    ndim = len(x_shape)
    if axis is None:
        axis = None
    elif not isinstance(axis, tuple):
        axis = (axis,)

    if not (ndim == 0 or axis is None or keepdims):
        # 处理负数索引
        actual_axis = [a if a >= 0 else a + ndim for a in axis]
        shape = list(gy.shape)
        for a in sorted(actual_axis):
            shape.insert(a, 1)
    else:
        shape = gy.shape

    gy = gy.reshape(shape)  # reshape
    return gy


def sum_to(x, shape):
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))

    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    return y
