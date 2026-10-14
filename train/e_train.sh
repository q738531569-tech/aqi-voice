#!/bin/bash
# E 组 v2 全自动流程：0.6B 底座 + 质心 embedding + 低音 oversample + lr 调度
# 云端跑法: nohup bash /root/e_data/e_train.sh > /root/e_log.txt 2>&1 < /dev/null &
set -e

# ---- 环境自适应 ----
if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
  source /root/miniconda3/etc/profile.d/conda.sh
  # 尝试旧实例保存的环境名（镜像里可能带着）
  for env in qwen-tts qwen aqi tts; do
    conda activate $env 2>/dev/null && break
  done
fi
python -c "import qwen_tts, torch, librosa" 2>/dev/null || {
  echo "[env] 缺依赖，pip 安装（清华源）"
  pip install -q -i https://pypi.tuna.tsinghua.edu.cn/simple qwen-tts librosa soundfile
}

# ---- repo 定位（镜像里可能已有；没有则 ghfast 克隆）----
REPO=""
for d in /root/EasyFinetuning /root/Qwen3-TTS-EasyFinetuning /root/autodl-tmp/EasyFinetuning /root/autodl-tmp/Qwen3-TTS-EasyFinetuning; do
  [ -f "$d/src/cli.py" ] && REPO=$d && break
done
if [ -z "$REPO" ]; then
  REPO=/root/EasyFinetuning
  git clone --depth 1 https://ghfast.top/https://github.com/mozi1924/Qwen3-TTS-EasyFinetuning.git $REPO
fi
echo "[repo] $REPO"
cd $REPO

mkdir -p tools raw-dataset
cp /root/e_data/tools/*.py tools/
cp /root/e_data/wavs/*.wav raw-dataset/
echo "[data] $(ls raw-dataset/*.wav | wc -l) 个 wav"

# ---- 1. split（切段 + 24k 重采样）----
python src/cli.py split --input_dir raw-dataset --speaker_name aqi_e

# ---- 2. asr（Qwen3-ASR 转写，ModelScope 源）----
python src/cli.py asr --speaker_name aqi_e --model_source ModelScope

# ---- 3. f0 扫描 + 低音 oversample（tokenize 之前！）----
python tools/jsonl_f0_oversample.py --jsonl final-dataset/aqi_e/tts_train.jsonl

# ---- 4. tokenize（复制段一并编码）----
python src/cli.py tokenize --speaker_name aqi_e --experiment_name e1 --model_source ModelScope

# ---- 5. embed avg_all（质心 embedding，低音天然加权）----
python src/cli.py embed --speaker_name aqi_e \
  --init_model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --mode avg_all --model_source ModelScope

# ---- 6. lr 调度补丁（warmup + cosine）----
python tools/patch_lr.py

# ---- 7. 训练（0.6B 底座提速 2.5x，lr 5e-7，5 epochs 逐 epoch 存档）----
python src/cli.py train --experiment_name e1 --speaker_name aqi_e \
  --init_model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --lr 5e-7 --epochs 5 --save_strategy epoch \
  --keep_last_n_checkpoints 6 --resume_from_checkpoint none \
  --model_source ModelScope

echo "E_TRAIN_DONE"
