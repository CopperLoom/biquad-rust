#!/usr/bin/env python3
"""
Run HSQ init's argmax on BOTH correction arrays (Rust-produced and AutoEQ-produced)
to determine whether the 0.08 dB correction-array difference shifts the argmax.
"""
import json, os
import numpy as np

DEBUG = os.path.join(os.path.dirname(__file__), '..', 'fixtures', 'debug')

def load(path):
    return json.load(open(path))

def hsq_argmax(freqs, target, min_fc=20, max_fc=10000):
    f = np.array(freqs)
    t = np.array(target)
    min_ix = int(np.sum(f < max(40, min_fc)))
    max_ix = int(np.sum(f < min(10000, max_fc)))
    cands = [abs(np.mean(t[i:])) for i in range(min_ix, max_ix)]
    argmax_pos = int(np.argmax(cands))
    # AutoEQ "bug": uses argmax_pos directly as freq index, no +min_ix
    fc_ix_autoeq_quirk = argmax_pos
    fc_quirk = f[fc_ix_autoeq_quirk]
    # The "correct" interpretation:
    fc_ix_correct = argmax_pos + min_ix
    fc_correct = f[fc_ix_correct]
    return {
        'min_ix': min_ix, 'max_ix': max_ix,
        'argmax_pos': argmax_pos,
        'fc_quirk_ix': fc_ix_autoeq_quirk, 'fc_quirk': fc_quirk,
        'fc_correct_ix': fc_ix_correct, 'fc_correct': fc_correct,
        'top5_pos': sorted(range(len(cands)), key=lambda i: -cands[i])[:5],
        'top5_vals': sorted(cands, reverse=True)[:5],
    }

for case in ['hexa__bass_heavy', 'origin_s__bright', 'origin_s__flat', 'zero2__bass_heavy']:
    print(f'\n=== {case} ===')
    rust   = load(f'{DEBUG}/{case}__correction_rust.json')
    autoeq = load(f'{DEBUG}/{case}__correction.json')
    freqs = [p['freq'] for p in rust]   # both have same grid
    rust_t = [p['db'] for p in rust]
    autoeq_t = [p['db'] for p in autoeq]

    r = hsq_argmax(freqs, rust_t)
    a = hsq_argmax(freqs, autoeq_t)

    print(f'  Rust  : argmax_pos={r["argmax_pos"]:3d}  fc_quirk={r["fc_quirk"]:.1f} Hz  fc_correct={r["fc_correct"]:.1f} Hz')
    print(f'  AutoEQ: argmax_pos={a["argmax_pos"]:3d}  fc_quirk={a["fc_quirk"]:.1f} Hz  fc_correct={a["fc_correct"]:.1f} Hz')
    print(f'  Rust top-5 argmax positions: {r["top5_pos"]} vals={[f"{v:.4f}" for v in r["top5_vals"]]}')
    print(f'  AutoEQ top-5 argmax positions: {a["top5_pos"]} vals={[f"{v:.4f}" for v in a["top5_vals"]]}')
    print(f'  min_ix={r["min_ix"]} max_ix={r["max_ix"]}')
