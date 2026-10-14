# -*- coding: utf-8 -*-
"""测量一批 wav 的中位基频（E 组 epoch 选型用）。用法: python measure_f0.py a.wav b.wav ..."""
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import librosa

for path in sys.argv[1:]:
    try:
        a, sr = librosa.load(path, sr=16000, mono=True)
        f0, _, _ = librosa.pyin(a, fmin=80, fmax=400, sr=sr)
        f0v = f0[~np.isnan(f0)]
        if len(f0v) < 10:
            print(f'{path}: no f0')
            continue
        print(f'{path}: f0_med={np.median(f0v):.0f}Hz '
              f'P10={np.percentile(f0v, 10):.0f} P90={np.percentile(f0v, 90):.0f} '
              f'dur={len(a) / sr:.1f}s', flush=True)
    except Exception as e:
        print(f'{path}: ERR {e}', flush=True)
