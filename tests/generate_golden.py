#!/usr/bin/env python3
"""
Generate golden files for biquad-rust integration tests.
Runs AutoEQ pipeline directly and writes JSON with Rust-native field names.

AutoEQ oracle: ~/claude/reference-implementations/autoEQ/
Pinned commit: 7ae0f56d53074872b028649617a22bbb4232feb7
"""

import json
import sys
import os
import numpy as np

sys.path.insert(0, os.path.expanduser('~/claude/reference-implementations/autoEQ'))

from autoeq.frequency_response import FrequencyResponse
from autoeq.peq import PEQ, Peaking, LowShelf, HighShelf
from autoeq.constants import PREAMP_HEADROOM, DEFAULT_FS

FIXTURES = os.path.join(os.path.dirname(__file__), 'fixtures')
GOLDEN_DIR = os.path.join(FIXTURES, 'golden')

IEMS = ['blessing3', 'hexa', 'andromeda', 'zero2', 'origin_s']
TARGETS = ['harman_ie_2019', 'diffuse_field', 'flat', 'v_shaped', 'bass_heavy', 'bright']

FILTER_TYPE_MAP = {
    'Peaking': 'PK',
    'LowShelf': 'LSQ',
    'HighShelf': 'HSQ',
}

# 3 constraint sets from PLAN.md §7
CONSTRAINT_SETS = {
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
    'restricted': {
        'filters': [
            {'type': 'PEAKING', 'min_gain': -6.0, 'max_gain': 6.0},
            {'type': 'PEAKING', 'min_gain': -6.0, 'max_gain': 6.0},
            {'type': 'PEAKING', 'min_gain': -6.0, 'max_gain': 6.0},
        ],
        'filter_defaults': {'min_q': 1.0, 'max_q': 5.0},
    },
    'qudelix_10': {
        'filters': (
            [{'type': 'LOW_SHELF',  'min_gain': -12.0, 'max_gain': 12.0}] +
            [{'type': 'PEAKING',    'min_gain': -12.0, 'max_gain': 12.0}] * 8 +
            [{'type': 'HIGH_SHELF', 'min_gain': -12.0, 'max_gain': 12.0}]
        ),
        'filter_defaults': {'min_q': 0.5, 'max_q': 10.0},
    },
}


def load_fr(path):
    with open(path) as f:
        data = json.load(f)
    freqs = np.array([p['freq'] for p in data])
    dbs = np.array([p['db'] for p in data])
    return FrequencyResponse(name='fr', frequency=freqs, raw=dbs)


def run_pipeline(iem_fr, target_fr, constraint_name, fs=DEFAULT_FS):
    meas = FrequencyResponse(name='meas', frequency=iem_fr.frequency.copy(), raw=iem_fr.raw.copy())
    meas.interpolate()
    meas.center()
    meas.compensate(target_fr)
    meas.equalize()

    config = dict(CONSTRAINT_SETS[constraint_name])
    # Deep copy filters list so repeated runs don't share state
    config['filters'] = [dict(f) for f in config['filters']]

    peqs = meas._optimize_peq_filters(config, fs)
    peq = peqs[0]  # single config produces single PEQ

    max_boost = peq.max_gain
    if max_boost > 0:
        pregain = -(max_boost + PREAMP_HEADROOM)
    else:
        pregain = 0.0

    filters = []
    for filt in peq.filters:
        filters.append({
            'filter_type': FILTER_TYPE_MAP[filt.__class__.__name__],
            'fc': float(filt.fc),
            'gain': float(filt.gain),
            'q': float(filt.q),
        })

    return {'pregain': float(pregain), 'filters': filters}


def main():
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    count = 0
    errors = []

    target_cache = {}
    for target_name in TARGETS:
        path = os.path.join(FIXTURES, 'targets', f'{target_name}.json')
        target_cache[target_name] = load_fr(path)

    for iem in IEMS:
        iem_path = os.path.join(FIXTURES, 'fr', f'{iem}.json')
        iem_fr = load_fr(iem_path)

        for target_name in TARGETS:
            target_fr = target_cache[target_name]

            for constraint_name in CONSTRAINT_SETS:
                out_name = f'{iem}__{target_name}__{constraint_name}.json'
                out_path = os.path.join(GOLDEN_DIR, out_name)
                try:
                    result = run_pipeline(iem_fr, target_fr, constraint_name)
                    result['iem'] = iem
                    result['target'] = target_name
                    result['constraint'] = constraint_name
                    result['fs'] = DEFAULT_FS
                    with open(out_path, 'w') as f:
                        json.dump(result, f, indent=2)
                    count += 1
                    print(f'  OK  {out_name}')
                except Exception as e:
                    errors.append((out_name, str(e)))
                    print(f'  ERR {out_name}: {e}', file=sys.stderr)

    print(f'\nGenerated {count} golden files.')
    if errors:
        print(f'{len(errors)} errors:', file=sys.stderr)
        for name, err in errors:
            print(f'  {name}: {err}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
