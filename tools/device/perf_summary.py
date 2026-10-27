"""Summarize an RKNN per-op performance table (from perfdetail): time per target (CPU/NPU),
CPU ops by type, slowest CPU ops.

Usage: python perf_summary.py <perf.txt>
"""

import re
import sys
from collections import defaultdict


def main():
    rows = []
    for line in open(sys.argv[1], encoding='utf-8', errors='replace'):
        m = re.match(r'^(\d+)\s+(\S+)\s+(\S+)\s+(CPU|NPU|GPU)\s+.*?\s(\d+/\d+/\d+)\s+(\d+)\s', line)
        if m:
            name = line.rstrip().split()[-1]
            rows.append((int(m.group(1)), m.group(2), m.group(4), int(m.group(6)), name))
    by_target = defaultdict(lambda: [0, 0])
    by_cpu_op = defaultdict(lambda: [0, 0])
    for _, op, tgt, us, _ in rows:
        by_target[tgt][0] += 1
        by_target[tgt][1] += us
        if tgt == 'CPU':
            by_cpu_op[op][0] += 1
            by_cpu_op[op][1] += us
    total = sum(v[1] for v in by_target.values())
    print(f'{len(rows)} ops, total {total / 1000:.1f} ms')
    for t, (n, us) in sorted(by_target.items()):
        print(f'  {t}: {n} ops, {us / 1000:.1f} ms ({100 * us / max(total, 1):.0f}%)')
    print('CPU ops by type:')
    for op, (n, us) in sorted(by_cpu_op.items(), key=lambda x: -x[1][1])[:15]:
        print(f'  {op:<22} {n:4d} ops {us / 1000:8.1f} ms')
    print('slowest CPU ops:')
    for r in sorted((r for r in rows if r[2] == 'CPU'), key=lambda r: -r[3])[:10]:
        print(f'  #{r[0]:<5} {r[1]:<20} {r[3] / 1000:6.2f} ms  {r[4][:70]}')


if __name__ == '__main__':
    main()
