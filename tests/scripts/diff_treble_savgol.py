#!/usr/bin/env python3
"""Diff Rust vs scipy for: error_in, savgol_normal, savgol_treble."""
import json, os

DEBUG = os.path.join(os.path.dirname(__file__), '..', 'fixtures', 'debug')

CASES = [
    ('hexa',     'bass_heavy'),
    ('origin_s', 'bright'),
    ('origin_s', 'flat'),
    ('zero2',    'bass_heavy'),
]
STAGES = ['error_in', 'savgol_normal', 'savgol_treble']

def load(p): return json.load(open(p))

def diff(a, b, label):
    if len(a) != len(b):
        print(f'  [{label}] LEN MISMATCH {len(a)} vs {len(b)}'); return
    diffs = [(i, p['freq'], p['db'] - q['db']) for i, (p, q) in enumerate(zip(a, b))]
    abs_d = [(i, f, d, abs(d)) for i, f, d in diffs]
    abs_d.sort(key=lambda t: -t[3])
    n = len(diffs)
    mean_a = sum(t[3] for t in abs_d) / n
    print(f'  [{label:14s}] n={n}  mean|Δ|={mean_a:.6f}  max|Δ|={abs_d[0][3]:.6f}  '
          f'>0.001:{sum(1 for t in abs_d if t[3]>0.001)}  '
          f'>0.01:{sum(1 for t in abs_d if t[3]>0.01)}')
    if abs_d[0][3] > 1e-5:
        print('    worst 5:')
        for i, f, d, _ in abs_d[:5]:
            print(f'      idx={i:3d}  f={f:8.2f}Hz  rust-py={d:+.6f}')

for iem, tgt in CASES:
    print(f'\n=== {iem}__{tgt} ===')
    base = f'{DEBUG}/{iem}__{tgt}__'
    for stage in STAGES:
        py = base + f'{stage}.json'
        rs = base + f'{stage}_rust.json'
        if not (os.path.exists(py) and os.path.exists(rs)):
            print(f'  [{stage}] missing'); continue
        diff(load(rs), load(py), stage)
