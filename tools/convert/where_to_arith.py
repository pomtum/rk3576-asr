"""Rewrite every Where(c, a, b) whose condition is a bool tensor into b + (a - b) * Cast(c, float),
so that it runs on the NPU (c is 0/1, so both forms are equal; a and b may be constants).

Usage: python where_to_arith.py <in.onnx> <out.onnx>
"""

import sys

import onnx
from onnx import TensorProto, helper, shape_inference


def main():
    src, dst = sys.argv[1], sys.argv[2]
    m = onnx.load(src)
    inferred = shape_inference.infer_shapes(m)
    elem = {v.name: v.type.tensor_type.elem_type for v in list(inferred.graph.value_info) + list(inferred.graph.input)}
    for n in m.graph.node:  # Constant 节点的输出类型
        if n.op_type == 'Constant':
            for a in n.attribute:
                if a.name == 'value':
                    elem[n.output[0]] = a.t.data_type
    for i in m.graph.initializer:
        elem[i.name] = i.data_type

    nodes, done, casts = [], 0, {}
    for n in m.graph.node:
        if n.op_type == 'Where' and elem.get(n.input[0]) == TensorProto.BOOL \
                and elem.get(n.input[1]) == TensorProto.FLOAT and elem.get(n.input[2]) == TensorProto.FLOAT:
            c, a, b = n.input
            if c not in casts:  # 同一个掩码只转换一次
                casts[c] = n.name + '_maskf'
                nodes.append(helper.make_node('Cast', [c], [casts[c]], name=n.name + '_cast', to=TensorProto.FLOAT))
            p = n.name
            nodes += [
                helper.make_node('Sub', [a, b], [p + '_diff'], name=p + '_sub'),
                helper.make_node('Mul', [p + '_diff', casts[c]], [p + '_masked'], name=p + '_mul'),
                helper.make_node('Add', [b, p + '_masked'], list(n.output), name=p + '_add'),
            ]
            done += 1
        else:
            nodes.append(n)
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    onnx.checker.check_model(m)
    onnx.save(m, dst)
    print(f'Where rewritten: {done}, remaining Where: {sum(1 for n in m.graph.node if n.op_type == "Where")}')


if __name__ == '__main__':
    main()
