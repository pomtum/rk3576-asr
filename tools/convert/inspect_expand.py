"""Show the producers and consumers of the Expand nodes in an encoder ONNX.

Usage: python inspect_expand.py <encoder.onnx>
"""

import sys

import onnx
from onnx import shape_inference

TYPES = {1: 'float', 6: 'int32', 7: 'int64', 9: 'bool', 10: 'float16'}


def main():
    m = shape_inference.infer_shapes(onnx.load(sys.argv[1]))
    info = {}
    for v in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output):
        t = v.type.tensor_type
        info[v.name] = (TYPES.get(t.elem_type, t.elem_type), [d.dim_value for d in t.shape.dim])
    prod = {o: n for n in m.graph.node for o in n.output}
    users = {}
    for n in m.graph.node:
        for i in n.input:
            users.setdefault(i, []).append(n)

    def chain(name, depth=6):
        out = []
        while name in prod and depth:
            n = prod[name]
            out.append(f'{n.op_type}({n.name})')
            name = n.input[0] if n.input else ''
            depth -= 1
        out.append(name if name in info and name not in prod else '')
        return ' <- '.join(out)

    for n in m.graph.node:
        if n.op_type in ('Expand', 'Tile') and ('Expand' in n.name):
            o = n.output[0]
            print(f'== {n.name} {n.op_type} out={info.get(o)}')
            print('   source:', chain(n.input[0]))
            for u in users.get(o, []):
                print(f'   used by {u.op_type}({u.name}) -> {info.get(u.output[0])}')


if __name__ == '__main__':
    main()
