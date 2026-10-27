"""List the remaining Where nodes of an encoder ONNX and where their inputs come from.

Usage: python inspect_where.py <encoder.onnx>
"""

import sys

import onnx
from onnx import numpy_helper, shape_inference

TYPES = {1: 'float', 7: 'int64', 9: 'bool', 10: 'f16'}


def main():
    m = shape_inference.infer_shapes(onnx.load(sys.argv[1]))
    info = {v.name: (TYPES.get(v.type.tensor_type.elem_type, v.type.tensor_type.elem_type),
                     [d.dim_value for d in v.type.tensor_type.shape.dim])
            for v in list(m.graph.value_info) + list(m.graph.input)}
    prod = {o: n for n in m.graph.node for o in n.output}
    consts = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    for n in m.graph.node:
        if n.op_type == 'Constant':
            for a in n.attribute:
                if a.name == 'value':
                    consts[n.output[0]] = numpy_helper.to_array(a.t)

    def d(name):
        if name in consts:
            v = consts[name]
            return f'const{list(v.shape)}={v.ravel()[:1]}'
        p = prod.get(name)
        return f'{p.op_type if p else "input"}{info.get(name, "")}'

    for n in m.graph.node:
        if n.op_type == 'Where':
            print(n.name, '| cond', d(n.input[0]), '| a', d(n.input[1]), '| b', d(n.input[2]), '->', info.get(n.output[0]))


if __name__ == '__main__':
    main()
