#!/usr/bin/env python3
"""
Diff Rust vs AutoEQ correction (equalization) arrays on both 1.01 and 1.02 grids.

Run from ~/claude/biquad-rust:
    python3 tests/scripts/diff_correction.py

Reports mean/max diffs and worst points per case.
Localizes the hexa__bass_heavy bug:
  - If 1.01 arrays match (~1e-4 dB) but 1.02 differ → bug is in 1.01→1.02 interpolation.
  - If 1.01 arrays already differ at ~0.004 dB → bug is upstream in equalize.
"""
import json, os, sys

DEBUG = os.path.join(os.path.dirname(__file__), '..', 'fixtures', 'debug')

CASES = [
    ('hexa',     'bass_heavy'),
    ('origin_s', 'bright'),
    ('origin_s', 'flat'),
    ('zero2',    'bass_heavy'),
    ('hexa',     'diffuse_field'),
    ('origin_s', 'bass_heavy'),
]

def load(path):
    return json.load(open(path))

def diff(a, b, label):
    if len(a) != len(b):
        print(f'  [{label}] LENGTH MISMATCH: {len(a)} vs {len(b)}')
        return
    diffs = [(i, p['freq'], p['db'] - q['db']) for i, (p, q) in enumerate(zip(a, b))]
    abs_diffs = [(i, f, d, abs(d)) for i, f, d in diffs]
    abs_diffs.sort(key=lambda t: -t[3])
    n = len(diffs)
    mean_abs = sum(t[3] for t in abs_diffs) / n
    max_abs  = abs_diffs[0][3]
    over_001 = sum(1 for t in abs_diffs if t[3] > 0.001)
    over_01  = sum(1 for t in abs_diffs if t[3] > 0.01)
    print(f'  [{label}] n={n}  mean|Δ|={mean_abs:.5f}  max|Δ|={max_abs:.5f}  '
          f'>0.001:{over_001}  >0.01:{over_01}')
    print(f'    worst 10:')
    for i, f, d, ad in abs_diffs[:10]:
        print(f'      idx={i:3d}  f={f:8.2f}Hz  rust-autoeq={d:+.5f}')

for iem, tgt in CASES:
    print(f'\n=== {iem}__{tgt} ===')
    base = f'{DEBUG}/{iem}__{tgt}'.rstrip('_')
    for label, suffix_a, suffix_b in [
        ('1.01', '__correction_rust_1p01.json', '__correction_1p01.json'),
        ('1.02', '__correction_rust.json',      '__correction.json'),
    ]:
        pa, pb = base + suffix_a, base + suffix_b
        if not (os.path.exists(pa) and os.path.exists(pb)):
            print(f'  [{label}] missing: {os.path.basename(pa) if not os.path.exists(pa) else os.path.basename(pb)}')
            continue
        diff(load(pa), load(pb), label)
