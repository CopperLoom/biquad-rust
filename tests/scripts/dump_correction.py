#!/usr/bin/env python3
"""
Dump AutoEQ's correction (equalization) array on the 1.02 optimizer grid
for the hexa+bass_heavy case — for diffing against our Rust output.

Run from ~/claude/biquad-rust:
    python3 tests/scripts/dump_correction.py
Writes tests/fixtures/debug/hexa__bass_heavy__correction.json
"""
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.expanduser('~/claude/reference-implementations/autoEQ'))

from autoeq.frequency_response import FrequencyResponse
from autoeq.constants import DEFAULT_BIQUAD_OPTIMIZATION_F_STEP

FIXTURES = os.path.join(os.path.dirname(__file__), '..', 'fixtures')
DEBUG_DIR = os.path.join(FIXTURES, 'debug')
os.makedirs(DEBUG_DIR, exist_ok=True)

CASES = [
    ('hexa',     'bass_heavy'),
    ('origin_s', 'bright'),
    ('origin_s', 'flat'),
    ('zero2',    'bass_heavy'),
    # New failing cases after interpolate-extrapolation fix:
    ('hexa',     'diffuse_field'),
    ('origin_s', 'bass_heavy'),
]

def load_json_fr(name, subdir):
    path = os.path.join(FIXTURES, subdir, f'{name}.json')
    data = json.load(open(path))
    return (np.array([p['freq'] for p in data]),
            np.array([p['db']   for p in data]))

for iem, target_name in CASES:
    iem_f, iem_db = load_json_fr(iem, 'fr')
    tgt_f, tgt_db = load_json_fr(target_name, 'targets')

    meas = FrequencyResponse(name='meas', frequency=iem_f, raw=iem_db)
    tgt  = FrequencyResponse(name='tgt',  frequency=tgt_f, raw=tgt_db)

    meas.interpolate()
    meas.center()
    meas.compensate(tgt)
    meas.equalize()

    # 1.01-grid equalize output (BEFORE 1.02 resample) — for bisecting equalize bug
    out_101 = [{'freq': float(f), 'db': float(d)}
               for f, d in zip(meas.frequency, meas.equalization)]
    outpath_101 = os.path.join(DEBUG_DIR, f'{iem}__{target_name}__correction_1p01.json')
    json.dump(out_101, open(outpath_101, 'w'), indent=2)
    print(f'Wrote {len(out_101)} points to {outpath_101} (1.01 grid)')

    # Interpolate to 1.02 optimizer grid (same as _optimize_peq_filters)
    opt_fr = FrequencyResponse(
        name='opt',
        frequency=meas.frequency.copy(),
        equalization=meas.equalization.copy()
    )
    opt_fr.interpolate(f_step=DEFAULT_BIQUAD_OPTIMIZATION_F_STEP)

    out = [{'freq': float(f), 'db': float(d)}
           for f, d in zip(opt_fr.frequency, opt_fr.equalization)]

    outpath = os.path.join(DEBUG_DIR, f'{iem}__{target_name}__correction.json')
    json.dump(out, open(outpath, 'w'), indent=2)
    print(f'Wrote {len(out)} points to {outpath}')
    print(f'  freq range: {opt_fr.frequency[0]:.1f} - {opt_fr.frequency[-1]:.1f} Hz')
    print(f'  db range:   {opt_fr.equalization.min():.3f} - {opt_fr.equalization.max():.3f} dB')
