# -*- coding: utf-8 -*-
"""E 组：扫 tts_train.jsonl 每段音频的 f0，低音段直接复制条目（oversample）。

一步到位：读 jsonl -> 逐条 pyin 扫 f0 -> 按阈值复制 -> 重写 jsonl（备份 .bak）。
复制规则：f0<=240 -> x3; 240<f0<=250 -> x2; 其余 x1。
用法（repo 根目录）: python tools/jsonl_f0_oversample.py --jsonl final-dataset/aqi_e/tts_train.jsonl
"""
import argparse
import json
import shutil
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import librosa
import os


def seg_f0(path, sr=16000):
    try:
        a, _ = librosa.load(path, sr=sr, mono=True)
    except Exception:
        return None
    if len(a) < int(sr * 0.4):
        return None
    f0, _, _ = librosa.pyin(a, fmin=80, fmax=400, sr=sr)
    f0v = f0[~np.isnan(f0)]
    if len(f0v) < 10:
        return None
    return float(np.median(f0v))


def dup_of(f0):
    if f0 is None:
        return 1
    # 目标：把训练分布中心从 252Hz 压到 ~228Hz（生死线 230Hz）
    if f0 <= 235:
        return 4
    if f0 <= 250:
        return 2
    return 1


def resolve_audio(entry, jsonl_dir):
    p = str(entry.get('audio', ''))
    for cand in (p, os.path.join(jsonl_dir, p),
                 os.path.join(jsonl_dir, 'audio_24k', os.path.basename(p))):
        if os.path.exists(cand):
            return cand
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl', required=True)
    args = ap.parse_args()

    jdir = os.path.dirname(os.path.abspath(args.jsonl))
    entries = [json.loads(line) for line in open(args.jsonl, encoding='utf-8') if line.strip()]
    print(f'段数: {len(entries)}', flush=True)

    f0s = []
    for i, e in enumerate(entries):
        wav = resolve_audio(e, jdir)
        f0 = seg_f0(wav) if wav else None
        f0s.append(f0)
        if (i + 1) % 25 == 0:
            print(f'  scan {i + 1}/{len(entries)}', flush=True)

    json.dump([{'audio': e.get('audio'), 'f0': f0} for e, f0 in zip(entries, f0s)],
              open(args.jsonl + '.f0.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)

    valid = [f for f in f0s if f is not None]
    if valid:
        arr = np.array(valid)
        print(f'f0 有效 {len(valid)}/{len(f0s)}  P10={np.percentile(arr, 10):.0f} '
              f'P50={np.percentile(arr, 50):.0f} P90={np.percentile(arr, 90):.0f}', flush=True)

    shutil.copy(args.jsonl, args.jsonl + '.bak')
    out, n_low = [], 0
    for e, f0 in zip(entries, f0s):
        k = dup_of(f0)
        if k > 1:
            n_low += 1
        out.extend([e] * k)
    with open(args.jsonl, 'w', encoding='utf-8') as f:
        for e in out:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')
    print(f'oversample: {len(entries)} -> {len(out)} 段（低音加权 {n_low} 段）', flush=True)


if __name__ == '__main__':
    main()
