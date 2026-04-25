#!/usr/bin/env python3
"""
Generate per-stage equalize expected outputs for Phase 4 integration tests.
Runs AutoEQ pipeline through equalize() and writes the resulting equalization
curve as JSON so Rust tests can compare without depending on ephemeral /tmp files.

AutoEQ oracle: ~/claude/reference-implementations/autoEQ/
Pinned commit: 7ae0f56d53074872b028649617a22bbb4232feb7

Output: tests/fixtures/phase4_equalize/{iem}__{target}.json
Format: [{"freq": ..., "db": ...}, ...]  (matches FR fixture format)
"""

import json
import sys
import os
import numpy as np

sys.path.insert(0, os.path.expanduser('~/claude/reference-implementations/autoEQ'))

from autoeq.frequency_response import FrequencyResponse

FIXTURES = os.path.join(os.path.dirname(__file__), 'fixtures')
OUT_DIR = os.path.join(FIXTURES, 'phase4_equalize')

IEMS = ['blessing3', 'hexa', 'andromeda', 'zero2', 'origin_s']
TARGETS = ['harman_ie_2019', 'diffuse_field', 'flat', 'v_shaped', 'bass_heavy', 'bright']


def load_fr(path):
    with open(path) as f:
        data = json.load(f)
    freqs = np.array([p['freq'] for p in data])
    dbs = np.array([p['db'] for p in data])
    return FrequencyResponse(name='fr', frequency=freqs, raw=dbs)


def run(iem, target_name):
    iem_fr = load_fr(os.path.join(FIXTURES, 'fr', f'{iem}.json'))
    target_fr = load_fr(os.path.join(FIXTURES, 'targets', f'{target_name}.json'))

    meas = FrequencyResponse(name='meas', frequency=iem_fr.frequency.copy(), raw=iem_fr.raw.copy())
    meas.interpolate()
    meas.center()
    meas.compensate(target_fr)
    meas.equalize()

    points = [
        {'freq': float(f), 'db': float(d)}
        for f, d in zip(meas.frequency, meas.equalization)
    ]
    return points


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    count = 0
    for iem in IEMS:
        for target in TARGETS:
            points = run(iem, target)
            out_path = os.path.join(OUT_DIR, f'{iem}__{target}.json')
            with open(out_path, 'w') as f:
                json.dump(points, f, indent=None, separators=(',', ':'))
            print(f'  {iem}__{target}: {len(points)} points')
            count += 1
    print(f'\nWrote {count} fixtures to {OUT_DIR}')


if __name__ == '__main__':
    main()
