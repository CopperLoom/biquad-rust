#!/usr/bin/env python3
"""
Diagnostic: intermediate-value comparison for 5 failing golden combinations.

For each failing case:
  1. AutoEQ equalization curve stats (the optimizer target)
  2. AutoEQ initial filter placements (from _init_optimizer_params)
  3. Golden (AutoEQ final) filter placements + fc spacing
  4. Loss of AutoEQ solution on the optimizer grid

Run from ~/claude/biquad-rust:
    python3 tests/scripts/debug_divergence.py
Then run:
    cargo test --test debug_intermediates -- --nocapture 2>&1 | grep -v warning
to see our Rust intermediates for the same cases.
"""
import json, sys, os, copy
import numpy as np
sys.path.insert(0, os.path.expanduser('~/claude/reference-implementations/autoEQ'))

from autoeq.frequency_response import FrequencyResponse
from autoeq.peq import PEQ, Peaking, LowShelf, HighShelf
from autoeq.constants import PREAMP_HEADROOM, DEFAULT_FS, DEFAULT_BIQUAD_OPTIMIZATION_F_STEP

FIXTURES  = os.path.join(os.path.dirname(__file__), '..', 'fixtures')
GOLDEN_DIR = os.path.join(FIXTURES, 'golden')

FAILING = [
    ('hexa',     'bass_heavy', 'qudelix_10'),
    ('hexa',     'bass_heavy', 'standard'),
    ('origin_s', 'bright',     'qudelix_10'),
    ('origin_s', 'flat',       'qudelix_10'),
    ('zero2',    'bass_heavy', 'restricted'),
]

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
FILTER_TYPE_MAP = {'Peaking': 'PK', 'LowShelf': 'LSQ', 'HighShelf': 'HSQ'}

def optimizer_grid():
    freqs, f = [], 20.0
    while f <= 20000.0:
        freqs.append(f); f *= 1.02
    return np.array(freqs)

def load_json_fr(name, subdir='fr'):
    path = os.path.join(FIXTURES, subdir, f'{name}.json')
    data = json.load(open(path))
    return (np.array([p['freq'] for p in data]),
            np.array([p['db']   for p in data]))

def make_fr(name, freqs, dbs):
    return FrequencyResponse(name=name, frequency=freqs, raw=dbs)

def load_golden(iem, target, constraint):
    return json.load(open(os.path.join(GOLDEN_DIR, f'{iem}__{target}__{constraint}.json')))

def eq_stats(label, freqs, vals):
    bands = [(20,200,'sub-bass'),(200,2000,'mid'),(2000,8000,'upper-mid'),(8000,20000,'treble')]
    parts = [f'{n}:[{vals[(freqs>=lo)&(freqs<=hi)].min():.2f},{vals[(freqs>=lo)&(freqs<=hi)].max():.2f}]'
             for lo,hi,n in bands if ((freqs>=lo)&(freqs<=hi)).any()]
    print(f'  {label}: {" ".join(parts)}')

def spacing_report(label, filters):
    pk_fcs = sorted([f['fc'] for f in filters if f['filter_type'] == 'PK'])
    if len(pk_fcs) < 2:
        return
    ratios = [pk_fcs[i+1]/pk_fcs[i] for i in range(len(pk_fcs)-1)]
    octs   = [np.log2(r) for r in ratios]
    print(f'  {label} PK fc (Hz): {[f"{fc:.0f}" for fc in pk_fcs]}')
    print(f'  {label} PK spacing (oct): {[f"{o:.2f}" for o in octs]}  min={min(octs):.2f}')
    if min(octs) < 0.5:
        print(f'  *** CLUSTER: minimum gap {min(octs):.2f} oct ***')

def run_autoeq_pipeline(iem, target_name, constraint_name):
    iem_f,    iem_db  = load_json_fr(iem,         'fr')
    tgt_f,    tgt_db  = load_json_fr(target_name, 'targets')

    meas = make_fr('meas', iem_f,  iem_db)
    tgt  = make_fr('tgt',  tgt_f,  tgt_db)

    meas.interpolate(); meas.center(); meas.compensate(tgt); meas.equalize()

    # Build the optimizer FR (interpolated to 1.02 grid) — same as _optimize_peq_filters
    opt_fr = FrequencyResponse(name='opt', frequency=meas.frequency.copy(),
                               equalization=meas.equalization.copy())
    opt_fr.interpolate(f_step=DEFAULT_BIQUAD_OPTIMIZATION_F_STEP)

    config = dict(CONSTRAINT_SETS[constraint_name])
    config['filters'] = copy.deepcopy(config['filters'])
    defaults = config.get('filter_defaults', {})

    # Get initial filter placements (init mutates filter objects)
    peq_init = PEQ.from_dict(config, opt_fr.frequency, DEFAULT_FS, target=opt_fr.equalization)
    peq_init._init_optimizer_params()   # mutates filter.fc/gain/q in place
    init_filters = [
        {'type': FILTER_TYPE_MAP[type(f).__name__], 'fc': float(f.fc),
         'gain': float(f.gain), 'q': float(f.q)}
        for f in peq_init.filters
    ]

    return {
        'eq_freqs': opt_fr.frequency,
        'eq_vals':  opt_fr.equalization,
        'init_filters': init_filters,
    }

def main():
    grid = optimizer_grid()

    for iem, target, constraint in FAILING:
        name = f'{iem}__{target}__{constraint}'
        print(f'\n{"="*68}')
        print(f'CASE: {name}')
        print(f'{"="*68}')

        golden = load_golden(iem, target, constraint)
        autoeq = run_autoeq_pipeline(iem, target, constraint)

        # 1. Equalization curve
        print('\n[1] AutoEQ optimizer target (equalization curve):')
        eq_stats('eq', autoeq['eq_freqs'], autoeq['eq_vals'])

        # 2. Initial filter placements
        print('\n[2] AutoEQ initial filter placements:')
        for f in autoeq['init_filters']:
            print(f'  {f["type"]:3s}  fc={f["fc"]:8.1f} Hz  gain={f["gain"]:7.3f} dB  q={f["q"]:.3f}')
        spacing_report('INIT', [{'filter_type': f['type'], 'fc': f['fc']}
                                 for f in autoeq['init_filters']])

        # 3. Golden (AutoEQ final) filters
        print('\n[3] Golden (AutoEQ final) filters:')
        for f in golden['filters']:
            print(f'  {f["filter_type"]:3s}  fc={f["fc"]:8.1f} Hz  gain={f["gain"]:7.3f} dB  q={f["q"]:.3f}')
        print(f'  pregain: {golden["pregain"]:.3f} dB')
        spacing_report('FINAL', golden['filters'])

        # 4. Loss of AutoEQ solution on optimizer grid
        from scipy.interpolate import interp1d
        eq_interp_fn = interp1d(autoeq['eq_freqs'], autoeq['eq_vals'],
                                 kind='linear', bounds_error=False,
                                 fill_value=(autoeq['eq_vals'][0], autoeq['eq_vals'][-1]))
        correction = eq_interp_fn(grid)

        # cascade filter responses on grid using scipy freqz
        from scipy.signal import freqz
        def cascade_db(filters, pregain):
            total = np.zeros(len(grid))
            fs = DEFAULT_FS
            for f in filters:
                fc, gain, q = f['fc'] if isinstance(f, dict) else f.fc, \
                               f['gain'] if isinstance(f, dict) else f.gain, \
                               f['q'] if isinstance(f, dict) else f.q
                ftype = f['filter_type'] if isinstance(f, dict) else FILTER_TYPE_MAP[type(f).__name__]
                w0 = 2*np.pi*fc/fs; A = 10**(gain/40); cos_w0 = np.cos(w0)
                if ftype == 'PK':
                    alpha = np.sin(w0)/(2*q)
                    b = [(1+alpha*A)/1, -2*cos_w0, 1-alpha*A]
                    a = [1+alpha/A,     -2*cos_w0, 1-alpha/A]
                elif ftype in ('LSQ','LOW_SHELF'):
                    alpha = np.sin(w0)/2*np.sqrt((A+1/A)*(1/0.7-1)+2)
                    b = [A*((A+1)-(A-1)*cos_w0+2*np.sqrt(A)*alpha),
                         2*A*((A-1)-(A+1)*cos_w0),
                         A*((A+1)-(A-1)*cos_w0-2*np.sqrt(A)*alpha)]
                    a = [(A+1)+(A-1)*cos_w0+2*np.sqrt(A)*alpha,
                         -2*((A-1)+(A+1)*cos_w0),
                         (A+1)+(A-1)*cos_w0-2*np.sqrt(A)*alpha]
                else:
                    alpha = np.sin(w0)/2*np.sqrt((A+1/A)*(1/0.7-1)+2)
                    b = [A*((A+1)+(A-1)*cos_w0+2*np.sqrt(A)*alpha),
                         -2*A*((A-1)+(A+1)*cos_w0),
                         A*((A+1)+(A-1)*cos_w0-2*np.sqrt(A)*alpha)]
                    a = [(A+1)-(A-1)*cos_w0+2*np.sqrt(A)*alpha,
                         2*((A-1)-(A+1)*cos_w0),
                         (A+1)-(A-1)*cos_w0-2*np.sqrt(A)*alpha]
                b = [x/a[0] for x in b]; a = [x/a[0] for x in a]
                _, h = freqz(b, a, worN=2*np.pi*grid/fs)
                total += 20*np.log10(np.maximum(np.abs(h), 1e-12))
            return total + pregain

        golden_cascade = cascade_db(golden['filters'], golden['pregain'])
        golden_loss = float(np.mean((golden_cascade - correction)**2))
        print(f'\n[4] AutoEQ solution loss (MSE on optimizer grid): {golden_loss:.6f}')
        print(f'    (Rust will print its loss and filters below — run debug_intermediates test)')

if __name__ == '__main__':
    main()
