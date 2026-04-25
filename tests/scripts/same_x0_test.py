#!/usr/bin/env python3
"""
Perfect test: feed AutoEQ's x0 to both scipy SLSQP and our Rust optimizer.
If they produce the same result → divergence is purely in initialization.
If they diverge → the SLSQP implementations themselves differ.

Usage:
    cd ~/claude/biquad-rust
    python3 tests/scripts/same_x0_test.py
Then:
    cargo test --test same_x0 -- --nocapture
"""
import json, sys, os, copy
import numpy as np
sys.path.insert(0, os.path.expanduser('~/claude/reference-implementations/autoEQ'))

from autoeq.frequency_response import FrequencyResponse
from autoeq.peq import PEQ, Peaking, LowShelf, HighShelf
from autoeq.constants import DEFAULT_FS, DEFAULT_BIQUAD_OPTIMIZATION_F_STEP

FIXTURES = os.path.join(os.path.dirname(__file__), '..', 'fixtures')

CONSTRAINT_SETS = {
    'qudelix_10': {
        'filters': (
            [{'type': 'LOW_SHELF',  'min_gain': -12.0, 'max_gain': 12.0}] +
            [{'type': 'PEAKING',    'min_gain': -12.0, 'max_gain': 12.0}] * 8 +
            [{'type': 'HIGH_SHELF', 'min_gain': -12.0, 'max_gain': 12.0}]
        ),
        'filter_defaults': {'min_q': 0.5, 'max_q': 10.0},
    },
    'standard': {
        'filters': [
            {'type': 'LOW_SHELF',  'min_gain': -12.0, 'max_gain': 12.0},
            {'type': 'PEAKING',    'min_gain': -12.0, 'max_gain': 12.0},
            {'type': 'PEAKING',    'min_gain': -12.0, 'max_gain': 12.0},
            {'type': 'PEAKING',    'min_gain': -12.0, 'max_gain': 12.0},
            {'type': 'HIGH_SHELF', 'min_gain': -12.0, 'max_gain': 12.0},
        ],
        'filter_defaults': {'min_q': 0.5, 'max_q': 10.0},
    },
}
FILTER_TYPE_MAP = {'Peaking': 'PK', 'LowShelf': 'LSQ', 'HighShelf': 'HSQ'}

def load_json_fr(name, subdir):
    data = json.load(open(os.path.join(FIXTURES, subdir, f'{name}.json')))
    return (np.array([p['freq'] for p in data]), np.array([p['db'] for p in data]))

def make_fr(name, f, db):
    return FrequencyResponse(name=name, frequency=f, raw=db)

def run_pipeline(iem, target_name, constraint_name):
    meas = make_fr('m', *load_json_fr(iem, 'fr'))
    tgt  = make_fr('t', *load_json_fr(target_name, 'targets'))
    meas.interpolate(); meas.center(); meas.compensate(tgt); meas.equalize()

    opt_fr = FrequencyResponse(name='o', frequency=meas.frequency.copy(),
                               equalization=meas.equalization.copy())
    opt_fr.interpolate(f_step=DEFAULT_BIQUAD_OPTIMIZATION_F_STEP)

    config = dict(CONSTRAINT_SETS[constraint_name])
    config['filters'] = copy.deepcopy(config['filters'])
    defaults = config.get('filter_defaults', {})

    peq = PEQ.from_dict(config, opt_fr.frequency, DEFAULT_FS, target=opt_fr.equalization)

    # Extract x0 BEFORE optimization
    x0 = peq._init_optimizer_params()
    init_filters = [(FILTER_TYPE_MAP[type(f).__name__], float(f.fc), float(f.gain), float(f.q))
                    for f in peq.filters]

    # Run AutoEQ's optimization from this x0
    peq.optimize()
    autoeq_filters = [(FILTER_TYPE_MAP[type(f).__name__], float(f.fc), float(f.gain), float(f.q))
                      for f in peq.filters]

    return {
        'x0': x0.tolist(),
        'init_filters': init_filters,
        'autoeq_filters': autoeq_filters,
        'opt_freqs': opt_fr.frequency.tolist(),
        'correction': opt_fr.equalization.tolist(),
        'fs': DEFAULT_FS,
    }

CASES = [
    ('hexa', 'bass_heavy', 'qudelix_10'),
    ('hexa', 'bass_heavy', 'standard'),
]

def main():
    os.makedirs(os.path.join(FIXTURES, 'debug'), exist_ok=True)
    for iem, target, constraint in CASES:
        name = f'{iem}__{target}__{constraint}'
        print(f'\n{"="*60}\nCASE: {name}\n{"="*60}')
        data = run_pipeline(iem, target, constraint)

        print(f'x0 ({len(data["x0"])} params): {[f"{v:.4f}" for v in data["x0"]]}')
        print('\nInit filters (AutoEQ):')
        for t, fc, gain, q in data['init_filters']:
            print(f'  {t:3s}  fc={fc:8.1f}  gain={gain:7.3f}  q={q:.3f}')
        print('\nFinal filters (AutoEQ after optimization from this x0):')
        for t, fc, gain, q in data['autoeq_filters']:
            print(f'  {t:3s}  fc={fc:8.1f}  gain={gain:7.3f}  q={q:.3f}')

        # Write x0 for Rust to consume
        out = os.path.join(FIXTURES, 'debug', f'{name}__x0.json')
        json.dump({'x0': data['x0'], 'init_filters': data['init_filters'],
                   'autoeq_filters': data['autoeq_filters']}, open(out, 'w'), indent=2)
        print(f'\nWrote: {out}')

if __name__ == '__main__':
    main()
