# aqi-voice — 阿七音色 · DSH 实时语音插件

把 [DeepSeek Harness](https://github.com/memef1f1y/dsh)（DSH）的「自动朗读」变成**真人级御姐音实时对话**：
Qwen3-TTS 微调音色 + faster-qwen3-tts（CUDA Graphs）推理 + 服务生命周期插件托管。

> ⚠️ 音色权重为抖音创作者「阿七」的声音微调，仅供学习/演示，**非商用**。详见 [VOICE-LICENSE.md](VOICE-LICENSE.md)。

## 效果

- 音色：真人级（听感采样见 `samples/`，`ep3_t2.wav` 为最佳示范）
- 速度：**RTF 0.37**（RTX 3060 Ti 8GB 实测，2.7 倍速于实时），首声 ~0.4s
- 资源：跟「自动朗读」开关走——开则后台常驻，关则进程退出、显存归零；DSH 退出服务自动退出
- 全程无窗口、无任务栏图标

## 架构

```
DSH「自动朗读」开关
   │ wake/sleep（伴生插件 aqi-tts-service 托管进程生死）
   ▼
qwen_aqi_server.py  :7890  ← CosyVoice 兼容 HTTP 协议
   │ faster-qwen3-tts（CUDA Graphs 手写解码步，launch-bound 归零）
   ▼
Qwen3-TTS-12Hz-0.6B 微调 checkpoint（speaker=aqi_e）
```

- `plugin/` —— DSH 伴生插件 `aqi-tts-service`：跟随朗读开关自动拉起/停止服务，崩溃 30s 自愈，
  端口探活防重复拉起（spawn 风暴免疫）
- `server/` —— CosyVoice 协议服务（`/health` `/tts` `/voices`），路径全部 env 可配
- `train/` —— 云端微调流水线脚本（数据 jsonl 过采样、lr 补丁、训练/推理、f0 测量）
- `persona.md` —— 阿七说话风格 prompt（音色+说话风格是一套，贴进系统提示即用）

## 快速开始（Windows + NVIDIA GPU，≥8GB 显存）

1. **装 torch**（按你的 CUDA 版本，官方或国内镜像均可）：
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```
2. **装依赖**（`transformers` 必须钉 5.15.1，5.19 会崩 `MimiConfig.rope_theta`）：
   ```bash
   pip install -r server/requirements.txt
   ```
3. **下载权重**：从 [Releases](../../releases) 下载 **两个** zip（GitHub 单文件 2G 限制故拆包）：
   `aqi-voice-model-main.zip` 与 `aqi-voice-model-speech-tokenizer.zip`，
   **都解压到 `server/model/`**（解压后目录里应直接可见 `model.safetensors` 与 `speech_tokenizer/`）。
4. **起服务**（或跳过，见下一步由插件自动拉起）：
   ```bash
   python server/qwen_aqi_server.py --preload
   ```
5. **装 DSH 插件**：
   ```bash
   dsh plugin add "file:./plugin"
   ```
   然后在 DSH 设置里：TTS provider 选 **CosyVoice**（地址 `http://127.0.0.1:7890`），
   打开「自动朗读」。伴生插件会自动管理服务进程。
6. 环境变量（可选）：`AQI_PYTHON`（服务 Python）、`AQI_SCRIPT`（服务脚本）、
   `AQI_CKPT` / `AQI_SPEAKER` / `AQI_PROMPTS` / `AQI_TMP`（服务侧）。

## 用自己的音色训练（train/）

1. 准备语料：10–30 分钟干净人声 → ASR 对齐 → Qwen3-TTS 官方 checkpoint 数据格式（jsonl）
2. `python train/jsonl_f0_oversample.py`：按音高分层过采样（低音段加权）
3. `python train/patch_lr.py` + `bash train/e_train.sh`：LoRA-free 全参微调（AutoDL 单卡 ~2h/epoch）
4. `bash train/e_infer.sh` + `python train/measure_f0.py`：多 epoch 出样、测中位音高挑 checkpoint
5. 把选中的 checkpoint 放到 `server/model/`，speaker 名同步 `AQI_SPEAKER`

## 已知坑

- `transformers` 5.19 与 qwen-tts-hf 不兼容 → 钉 **5.15.1**
- Electron 宿主环境的 TEMP 可能不可用 → 服务开头写死专属 temp（脚本已内置）
- 句级分块会导致**块间音色漂移** → 整段生成（RTF 够快，不需要分块）
- 采样温度调低音高不可行（实测 0.3 出气声）→ 低音靠 instruct 指令
- 本服务的 `best_of`（抽卡挑最低音）可被 `/tts` 请求体覆盖；dsh-plugin-tts 上游暂无此设置项，
  本仓库用法是在其 CosyVoice 面板加「抽卡次数」输入（patch `client.js`/`index.mjs` 各一处，向 `best_of` 字段透传即可）

## License

- 代码：[MIT](LICENSE)
- 音色权重：[CC BY-NC-ND 4.0 + 免责声明](VOICE-LICENSE.md)
- Base model：Qwen3-TTS，Apache 2.0
