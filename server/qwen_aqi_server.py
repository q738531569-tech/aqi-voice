# -*- coding: utf-8 -*-
"""阿七音色服务 —— CosyVoice 协议兼容（cosy-server.py 的 Qwen3-TTS 平替）。

dsh-plugin-tts（provider=cosyvoice）只依赖四个端点：
  GET  /health   -> {"ok": true, "model_loaded": bool, ...}
  GET  /voices   -> {"voices": [文件名, ...]}          提示音频列表
  POST /upload   -> 保存提示音频（raw body 或 multipart）
  POST /tts      -> body {"text","prompt","prompt_text","speed","seed"}
                    返回 {"audio_base64": wav的base64, "sample_rate": int}

模型：_aqi/model/aqi_C_epoch4（Qwen3-TTS 微调定稿，speaker=aqi_c）。
定稿模型音色写死在 embedding 里，prompt 仅作占位（插件 UI 需要选一个）。
空闲 IDLE_UNLOAD_SEC 秒自动卸载模型（显存归零），下次请求自动重载。

用法：python qwen_aqi_server.py            （默认 127.0.0.1:7890）
      python qwen_aqi_server.py --port 7890 --no-idle
"""
import argparse
import base64
import io
import json
import os
import sys
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))

# 宿主环境（Electron 插件 host 等）的 TEMP 可能不可用，torch inductor
# 初始化会崩 —— 写死一个必定可用的专属 temp
_TMP = os.environ.get('AQI_TMP') or os.path.join(HERE, 'tmp')
os.makedirs(_TMP, exist_ok=True)
for _k in ('TMP', 'TEMP', 'TMPDIR'):
    os.environ[_k] = _TMP

# 模型与音色：env 可配，默认仓库内 server/model（从 Release 下载解压到此）
CKPT = os.environ.get('AQI_CKPT') or os.path.join(HERE, 'model')
PROMPT_DIR = os.environ.get('AQI_PROMPTS') or os.path.join(HERE, 'prompts')
SPEAKER = os.environ.get('AQI_SPEAKER', 'aqi_e')
LANGUAGE = 'chinese'
IDLE_UNLOAD_SEC = 600  # 空闲 10 分钟卸载模型，显存归零

_lock = threading.Lock()      # GPU 串行锁（同时保护 _model 读写）
_model = None
_last_used = time.time()
_idle_enabled = True


LOG_FILE = os.path.join(_TMP, 'qwen_aqi_server.log')  # Electron host 子进程写不了 tools，tmp 实测可写


def log(msg):
    line = f'[aqi-tts] {msg}'
    print(line, flush=True)
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f'{time.strftime("%H:%M:%S")} {line}\n')
    except OSError:
        pass


def load_model():
    global _model
    if _model is not None:
        return
    import torch  # noqa: F401  确认可用再加载
    from faster_qwen3_tts import FasterQwen3TTS
    t0 = time.time()
    log(f'loading model from {CKPT} ...')
    # faster-qwen3-tts：手写 CUDA Graphs，RTF 0.49（旧实现 2.6+），需 py-faster 环境
    _model = FasterQwen3TTS.from_pretrained(CKPT)
    _model.warmup(prefill_len=100)
    log(f'model loaded in {time.time() - t0:.1f}s')


def unload_model():
    global _model
    if _model is None:
        return
    _model = None
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    log('model unloaded (idle), VRAM released')


def idle_watchdog():
    while True:
        time.sleep(30)
        global _last_used
        if (_idle_enabled and _model is not None
                and time.time() - _last_used > IDLE_UNLOAD_SEC):
            with _lock:
                if time.time() - _last_used > IDLE_UNLOAD_SEC:
                    unload_model()


BEST_OF_DEFAULT = 2  # 双候选挑最低音（/tts 可传 best_of 覆盖）；TARGET_F0 提前收手兜底
TARGET_F0 = 230.0    # 生死线（≤226Hz 像她）：首发达标即收手，平均省 ~30% 时间
# 插件 UI 没有 instruct 字段，服务端内置默认压低指令（免费降 ~15Hz）
DEFAULT_INSTRUCT = '用低沉磁性的御姐声线说话，音调刻意压低'


TARGET_RMS = 0.075   # 响度锁定目标（≈-22.5 dBFS）：治模型音量忽大忽小


def _pcm16_wav(audio_f32, sr):
    import numpy as np
    audio = np.asarray(audio_f32, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    if rms > 1e-6:
        audio = audio * (TARGET_RMS / rms)
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype('<i2')
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _median_f0(wav_bytes):
    """检测 wav 的中位音高（Hz），失败返回 +inf（不挑它）。"""
    try:
        import numpy as np
        import librosa
        y, sr = librosa.load(io.BytesIO(wav_bytes), sr=16000)
        # 只测中段 70%：首尾静音/气声既干扰测量又拖慢 pyin
        y = y[int(len(y) * 0.15):int(len(y) * 0.85)]
        f0, vflag, _ = librosa.pyin(y, fmin=80, fmax=500, frame_length=1024)
        f0v = f0[~np.isnan(f0)]
        if len(f0v) < 10:
            return float('inf')
        return float(np.median(f0v))
    except Exception:  # noqa: BLE001
        return float('inf')


def synth_wav(text, seed=0, best_of=BEST_OF_DEFAULT, instruct=None, temperature=None):
    """合成一句文本，返回 (wav_bytes, sample_rate)。

    best_of>1 时合成多条，自动挑中位音高最低的（用户偏好：越低沉越像御姐）。
    """
    global _last_used
    best_of = max(1, int(best_of or 1))
    t0 = time.time()
    best = (float('inf'), None, 0)  # (f0, wav_bytes, sr)
    for i in range(best_of):
        with _lock:
            load_model()
            _last_used = time.time()
            import torch
            if seed:
                # 指定 seed：第一条用该 seed，挑优时递增
                torch.manual_seed(int(seed) + i)
            kwargs = {}
            if instruct:
                kwargs['instruct'] = str(instruct)
            if temperature:
                kwargs['temperature'] = float(temperature)
            wavs, sr = _model.generate_custom_voice(
                text=text, speaker=SPEAKER, language=LANGUAGE, **kwargs)
            _last_used = time.time()
        wav_bytes = _pcm16_wav(wavs[0], sr)
        if best_of == 1:
            best = (0.0, wav_bytes, int(sr))
            break
        f0 = _median_f0(wav_bytes)
        log(f'  candidate {i + 1}/{best_of}: median f0={f0:.0f}Hz')
        if best[1] is None or f0 < best[0]:
            best = (f0, wav_bytes, int(sr))
        if f0 <= TARGET_F0:
            # 已抽中低音版（≤生死线），不再抽第二发——省一半时间
            break
    log(f'synth {len(text)} chars best_of={best_of} in {time.time() - t0:.1f}s'
        f' -> chosen f0={best[0]:.0f}Hz {len(best[1])} bytes')
    return best[1], best[2]


def list_voices():
    if not os.path.isdir(PROMPT_DIR):
        return []
    exts = ('.wav', '.mp3', '.flac', '.m4a', '.ogg', '.opus')
    return sorted(f for f in os.listdir(PROMPT_DIR)
                  if f.lower().endswith(exts))


def save_upload(body, content_type, fallback_name):
    """保存上传的提示音频；支持 multipart/form-data 与 raw body。返回文件名。"""
    os.makedirs(PROMPT_DIR, exist_ok=True)
    name = None
    data = body
    if 'multipart/form-data' in (content_type or ''):
        # 粗略 multipart 解析：找第一个带 filename 的 part，取其内容
        boundary = None
        for seg in content_type.split(';'):
            seg = seg.strip()
            if seg.startswith('boundary='):
                boundary = seg.split('=', 1)[1].strip('"')
        if boundary:
            marker = ('--' + boundary).encode()
            parts = body.split(marker)
            for p in parts:
                if b'filename=' not in p:
                    continue
                head, _, payload = p.partition(b'\r\n\r\n')
                for h in head.split(b'\r\n'):
                    if b'filename=' in h:
                        try:
                            name = h.decode('utf-8', 'replace').split('filename="', 1)[1].split('"', 1)[0]
                        except IndexError:
                            pass
                payload = payload.rstrip(b'\r\n-')
                if payload:
                    data = payload
                    break
    if not name:
        name = fallback_name or 'uploaded.wav'
    name = os.path.basename(name)
    if not name.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.ogg', '.opus')):
        name += '.wav'
    path = os.path.join(PROMPT_DIR, name)
    with open(path, 'wb') as f:
        f.write(data)
    log(f'uploaded prompt: {name} ({len(data)} bytes)')
    return name


class Handler(BaseHTTPRequestHandler):
    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        n = int(self.headers.get('Content-Length') or 0)
        return self.rfile.read(n) if n else b''

    def do_GET(self):
        path = self.path.split('?')[0]
        if path == '/health':
            self._json(200, {'ok': True, 'engine': 'qwen3-tts-aqi',
                             'model_loaded': _model is not None,
                             'speaker': SPEAKER})
        elif path == '/voices':
            self._json(200, {'voices': list_voices()})
        else:
            self._json(404, {'error': 'not found'})

    def do_POST(self):
        path = self.path.split('?')[0]
        body = self._read_body()
        if path == '/upload':
            try:
                name = save_upload(body, self.headers.get('Content-Type'),
                                   self.headers.get('X-Filename'))
                self._json(200, {'ok': True, 'name': name,
                                 'voices': list_voices()})
            except Exception as e:  # noqa: BLE001
                self._json(500, {'ok': False, 'message': str(e)})
        elif path == '/tts':
            try:
                req = json.loads(body.decode('utf-8')) if body else {}
            except json.JSONDecodeError:
                self._json(400, {'ok': False, 'message': 'bad json'})
                return
            text = str(req.get('text') or '').strip()
            if not text:
                self._json(400, {'ok': False, 'message': 'empty text'})
                return
            seed = int(req.get('seed') or 0)
            best_of = int(req.get('best_of') or BEST_OF_DEFAULT)
            instruct = req.get('instruct')
            if instruct is None:
                instruct = DEFAULT_INSTRUCT
            temperature = req.get('temperature') or None
            try:
                wav_bytes, sr = synth_wav(text, seed, best_of, instruct, temperature)
                self._json(200, {
                    'ok': True,
                    'audio_base64': base64.b64encode(wav_bytes).decode('ascii'),
                    'sample_rate': sr,
                })
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                self._json(500, {'ok': False, 'message': str(e)})
        elif path == '/load':
            # 兼容 RVC 风格预热调用
            try:
                load_model()
                self._json(200, {'ok': True})
            except Exception as e:  # noqa: BLE001
                self._json(500, {'ok': False, 'message': str(e)})
        else:
            self._json(404, {'error': 'not found'})

    def log_message(self, fmt, *args):  # 静音默认 access log
        pass


def main():
    global _idle_enabled
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=7890)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--no-idle', action='store_true', help='禁用空闲自动卸载')
    ap.add_argument('--preload', action='store_true', help='启动即加载模型')
    args = ap.parse_args()
    _idle_enabled = not args.no_idle

    if not os.path.isdir(CKPT):
        log(f'ERROR: checkpoint not found: {CKPT}')
        sys.exit(1)
    os.makedirs(PROMPT_DIR, exist_ok=True)
    log(f'ckpt={CKPT}')
    log(f'prompts={PROMPT_DIR} ({len(list_voices())} voices)')

    if args.preload:
        load_model()
    threading.Thread(target=idle_watchdog, daemon=True).start()

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    log(f'serving http://{args.host}:{args.port} (CosyVoice-compatible)')
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        log('bye')


if __name__ == '__main__':
    main()
