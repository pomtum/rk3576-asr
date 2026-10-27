"""Show the graph context of Equal / Max nodes (the ops RKNN runs on the CPU).

Usage: python inspect_fallback.py <encoder.onnx>
"""

import sys

import numpy as np
import onnx
from onnx import numpy_helper


def main():
    m = onnx.load(sys.argv[1])
    prod = {o: n for n in m.graph.node for o in n.output}
    users = {}
    for n in m.graph.node:
        for i in n.input:
            users.setdefault(i, []).append(n)
    consts = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    for n in m.graph.node:
        if n.op_type == 'Constant':
            for a in n.attribute:
                if a.name == 'value':
                    consts[n.output[0]] = numpy_helper.to_array(a.t)

    def desc(name):
        if name in consts:
            v = consts[name]
            return f'const{list(v.shape)}={v.ravel()[:2]}'
        p = prod.get(name)
        if p is None:
            return 'graph-input'
        ins = ', '.join(prod[i].op_type if i in prod and prod[i].op_type != 'Constant' else
                        (f'c={consts[i].ravel()[:1]}' if i in consts else 'in') for i in p.input)
        return f'{p.op_type}({ins})'

    shown = {'Equal': 0, 'Max': 0}
    for n in m.graph.node:
        if n.op_type in shown and shown[n.op_type] < 3:
            shown[n.op_type] += 1
            print(f'== {n.op_type} {n.name}')
            for i in n.input:
                print('   in:', desc(i))
            for u in users.get(n.output[0], []):
                print(f'   used by {u.op_type} {u.name} inputs:', [desc(i) for i in u.input])
    print('counts:', {t: sum(1 for n in m.graph.node if n.op_type == t) for t in ('Equal', 'Max', 'Where', 'Exp', 'Log')})


if __name__ == '__main__':
    main()
