"""Rewrite encoder ops that RKNN would run on the CPU into mathematically equivalent NPU ops:
  1. Max(0, x)                                         -> Relu(x)
  2. Where(Log(1+Exp(X)) == inf, X, Log(1+Exp(X)))     -> Relu(X) + Log(1 + Exp(-Abs(X)))   (stable softplus)
Unused nodes are removed afterwards.

Usage: python cpu_fallback_rewrite.py <in.onnx> <out.onnx>
"""

import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper


def main():
    src, dst = sys.argv[1], sys.argv[2]
    m = onnx.load(src)
    g = m.graph
    prod = {o: n for n in g.node for o in n.output}
    consts = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    for n in g.node:
        if n.op_type == 'Constant':
            for a in n.attribute:
                if a.name == 'value':
                    consts[n.output[0]] = numpy_helper.to_array(a.t)

    def is_const(name, value):
        v = consts.get(name)
        return v is not None and v.size == 1 and (np.isinf(value) and np.isinf(v.ravel()[0]) and v.ravel()[0] > 0
                                                  or v.ravel()[0] == value)

    one = 'cfr_one'
    g.initializer.append(numpy_helper.from_array(np.array(1.0, np.float32), one))
    replace, n_max, n_sp = {}, 0, 0
    for n in g.node:
        if n.op_type == 'Max' and len(n.input) == 2:
            a, b = n.input
            x = b if is_const(a, 0.0) else a if is_const(b, 0.0) else None
            if x is not None:
                replace[n.name] = [helper.make_node('Relu', [x], list(n.output), name=n.name + '_relu')]
                n_max += 1
        if n.op_type == 'Where':
            cond, x, lsum = n.input
            eq = prod.get(cond)
            if not eq or eq.op_type != 'Equal':
                continue
            l_in = [i for i in eq.input if not is_const(i, np.inf)]
            if len(l_in) != 1 or l_in[0] != lsum:
                continue
            log = prod.get(lsum)
            add = prod.get(log.input[0]) if log is not None and log.op_type == 'Log' else None
            if add is None or add.op_type != 'Add':
                continue
            exp_in = [prod[i] for i in add.input if i in prod and prod[i].op_type == 'Exp']
            others = [i for i in add.input if not (i in prod and prod[i].op_type == 'Exp')]
            if len(exp_in) != 1 or exp_in[0].input[0] != x or len(others) != 1 or not is_const(others[0], 1.0):
                print('skip unmatched', n.name)
                continue
            p = n.name
            replace[n.name] = [
                helper.make_node('Relu', [x], [p + '_relu'], name=p + '_relu'),
                helper.make_node('Abs', [x], [p + '_abs'], name=p + '_abs'),
                helper.make_node('Neg', [p + '_abs'], [p + '_neg'], name=p + '_neg'),
                helper.make_node('Exp', [p + '_neg'], [p + '_exp'], name=p + '_exp'),
                helper.make_node('Add', [p + '_exp', one], [p + '_add1'], name=p + '_add1'),
                helper.make_node('Log', [p + '_add1'], [p + '_log'], name=p + '_log'),
                helper.make_node('Add', [p + '_relu', p + '_log'], list(n.output), name=p + '_sp'),
            ]
            n_sp += 1

    nodes = []
    for n in g.node:
        nodes.extend(replace.get(n.name, [n]))
    # 删除不再被使用的节点（反复直到稳定）
    outputs = {o.name for o in g.output}
    while True:
        used = set(outputs)
        for n in nodes:
            used.update(n.input)
        keep = [n for n in nodes if any(o in used for o in n.output)]
        if len(keep) == len(nodes):
            break
        nodes = keep
    del g.node[:]
    g.node.extend(nodes)
    onnx.checker.check_model(m)
    onnx.save(m, dst)
    count = lambda t: sum(1 for n in g.node if n.op_type == t)
    print(f'Max->Relu: {n_max}, softplus rewritten: {n_sp}; remaining Equal={count("Equal")} Max={count("Max")} '
          f'Where={count("Where")}')


if __name__ == '__main__':
    main()
