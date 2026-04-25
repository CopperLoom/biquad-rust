#!/usr/bin/env python3
"""
Dump scipy.signal.savgol_filter (mode='interp', poly=2) over AutoEQ's compensate-error,
on the 1.01 grid, for both the 1/12-oct (normal) and 2-oct (treble) windows.

Also dumps the raw error array so we can compare error_in to Rust's.

Run from ~/claude/biquad-rust:
    python3 tests/scripts/dump_treble_savgol.py
"""
import json, os, sys
import numpy as np
from scipy.signal import savgol_filter
sys.path.insert(0, os.path.expanduser('~/claude/reference-implementations/autoEQ'))

from autoeq.frequency_response import FrequencyResponse
from autoeq.utils import smoothing_window_size

FIXTURES = os.path.join(os.path.dirname(__file__), '..', 'fixtures')
DEBUG = os.path.join(FIXTURES, 'debug')
os.makedirs(DEBUG, exist_ok=True)

CASES = [
    ('hexa',     'bass_heavy'),
    ('origin_s', 'bright'),
    ('origin_s', 'flat'),
    ('zero2',    'bass_heavy'),
]

def load_json_fr(name, subdir):
    path = os.path.join(FIXTURES, subdir, f'{name}.json')
    data = json.load(open(path))
    return (np.array([p['freq'] for p in data]),
            np.array([p['db']   for p in data]))

def write_pts(freqs, vals, path):
    out = [{'freq': float(f), 'db': float(v)} for f, v in zip(freqs, vals)]
    json.dump(out, open(path, 'w'), indent=2)

for iem, tgt in CASES:
    iem_f, iem_db = load_json_fr(iem, 'fr')
    tgt_f, tgt_db = load_json_fr(tgt,  'targets')

    meas = FrequencyResponse(name='meas', frequency=iem_f, raw=iem_db)
    tg   = FrequencyResponse(name='tg',   frequency=tgt_f, raw=tgt_db)
    meas.interpolate()
    meas.center()
    meas.compensate(tg)

    freqs   = meas.frequency
    err     = meas.error
    win_norm = smoothing_window_size(freqs, 1/12)
    win_treb = smoothing_window_size(freqs, 2)
    y_norm  = savgol_filter(err, win_norm, 2)  # mode='interp' is scipy default
    y_treb  = savgol_filter(err, win_treb, 2)

    print(f'{iem}__{tgt}: n={len(err)} win_normal={win_norm} win_treble={win_treb}')

    for suffix, arr in [('error_in', err), ('savgol_normal', y_norm), ('savgol_treble', y_treb)]:
        path = os.path.join(DEBUG, f'{iem}__{tgt}__{suffix}.json')
        write_pts(freqs, arr, path)
        print(f'  wrote {path}')
