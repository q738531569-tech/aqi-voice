# -*- coding: utf-8 -*-
"""给 sft_12hz.py 打 lr 调度补丁：DummyLRScheduler -> warmup+cosine。

理由（baseten 博客）：flat lr 早期过冲、晚期过拟合；warmup+cosine 音色质量更好。
幂等：已打过补丁则跳过。备份 .bak-lr。
用法（repo 根目录）: python tools/patch_lr.py
"""
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PATH = 'src/sft_12hz.py'
OLD = 'lr_scheduler = DummyLRScheduler()'
NEW = '''    # [E-group patch] warmup + cosine decay (baseten recipe)
    import math as _math
    from torch.optim.lr_scheduler import LambdaLR as _LambdaLR
    _steps_per_epoch = max(1, len(train_data) // max(1, args.batch_size * args.gradient_accumulation_steps))
    _total_steps = max(10, _steps_per_epoch * args.num_epochs)
    _warmup_steps = max(5, int(_total_steps * 0.05))

    def _warmup_cosine(cur):
        if cur < _warmup_steps:
            return float(cur) / float(max(1, _warmup_steps))
        p = float(cur - _warmup_steps) / float(max(1, _total_steps - _warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + _math.cos(_math.pi * min(1.0, p)))

    lr_scheduler = _LambdaLR(optimizer, _warmup_cosine)
    log_print(f"[E-patch] lr warmup={_warmup_steps} total={_total_steps} cosine decay")'''


def main():
    src = open(PATH, encoding='utf-8').read()
    if '[E-group patch] warmup' in src:
        print('已打过补丁，跳过')
        return
    if OLD not in src:
        print('!! 未找到锚点 "lr_scheduler = DummyLRScheduler()"，仓库结构可能已变')
        sys.exit(1)
    open(PATH + '.bak-lr', 'w', encoding='utf-8').write(src)
    open(PATH, 'w', encoding='utf-8').write(src.replace(OLD, NEW, 1))
    print('补丁完成: warmup(5%) + cosine 衰减已启用')


if __name__ == '__main__':
    main()
