"""Rewrite "broadcast a bool mask with Expand/Tile, then Where" into float arithmetic:
    Where(Expand(m), a, b)  ->  b + (a - b) * Cast(m, float)    (m keeps its small shape; Mul broadcasts)

On RK3576 the NPU mis-executes the bool Expand in the zipformer2 conv modules: models compiled with
rknn-toolkit2 2.2.0 output inf, models compiled with 2.3.2 hang the NPU (job timeout + soft reset).
Other Expand nodes (float upsampling) are kept.

Usage: python mask_where_to_mul.py <in.onnx> <out.onnx>
"""

import sys

import onnx
from onnx import TensorProto, helper, shape_inference


def main():
    src, dst = sys.argv[1], sys.argv[2]
    m = onnx.load(src)
    inferred = shape_inference.infer_shapes(m)
    elem = {v.name: v.type.tensor_type.elem_type
            for v in list(inferred.graph.value_info) + list(inferred.graph.input)}
    users = {}
    for n in m.graph.node:
        for i in n.input:
            users.setdefault(i, []).append(n)

    drop, replace = set(), {}
    for n in m.graph.node:
        if n.op_type not in ('Expand', 'Tile') or elem.get(n.output[0]) != TensorProto.BOOL:
            continue
        us = users.get(n.output[0], [])
        if not us or any(u.op_type != 'Where' or u.input[0] != n.output[0] for u in us):
            print('skip', n.name, [u.op_type for u in us])
            continue
        mask_f = n.name + '_maskf'
        replace[n.name] = [helper.make_node('Cast', [n.input[0]], [mask_f], name=n.name + '_cast', to=TensorProto.FLOAT)]
        for u in us:
            a, b = u.input[1], u.input[2]
            p = u.name
            replace[u.name] = [
                helper.make_node('Sub', [a, b], [p + '_diff'], name=p + '_sub'),
                helper.make_node('Mul', [p + '_diff', mask_f], [p + '_masked'], name=p + '_mul'),
                helper.make_node('Add', [b, p + '_masked'], list(u.output), name=p + '_add'),
            ]
        print(f'{n.name}: mask {elem.get(n.input[0])} -> {len(us)} Where rewritten')

    new_nodes = []
    for n in m.graph.node:
        new_nodes.extend(replace.get(n.name, [n]))
    del m.graph.node[:]
    m.graph.node.extend(new_nodes)
    meta = {}
    for p in m.metadata_props:
        meta.setdefault(p.key, p.value)
    del m.metadata_props[:]
    for k, v in meta.items():
        m.metadata_props.add(key=k, value=v)
    onnx.checker.check_model(m)
    onnx.save(m, dst)
    left = sum(1 for n in m.graph.node if n.op_type in ('Expand', 'Tile'))
    print(f'remaining Expand/Tile: {left}')


if __name__ == '__main__':
    main()
