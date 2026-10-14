#!/bin/bash
# E 组逐 epoch 试听 + f0 客观测量。跑完 e_train.sh 后执行：
#   nohup bash /root/e_data/e_infer.sh > /root/e_infer_log.txt 2>&1 < /dev/null &
set -e
REPO=${REPO:-/root/EasyFinetuning}
cd $REPO
mkdir -p /root/e_test

TEXT1='报告宝宝！今天姐姐带你逛吃逛吃呀，乖乖等姐姐一下下好不好～'
TEXT2='笨蛋宝宝，这么晚还不睡呀，姐姐给你唱首歌哄你，不许说姐姐幼稚哦。'

for ep in 1 2 3 4 5; do
  CK=output/e1/checkpoint-epoch-$ep
  [ -d "$CK" ] || { echo "skip $CK"; continue; }
  python src/cli.py infer --checkpoint $CK --speaker aqi_e --language Chinese \
    --text "$TEXT1" --output /root/e_test/ep${ep}_t1.wav
  python src/cli.py infer --checkpoint $CK --speaker aqi_e --language Chinese \
    --text "$TEXT2" --output /root/e_test/ep${ep}_t2.wav
done

python /root/e_data/tools/measure_f0.py /root/e_test/*.wav
echo "E_INFER_DONE"
