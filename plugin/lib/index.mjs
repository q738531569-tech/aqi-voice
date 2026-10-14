// aqi-tts-service —— Host 半（Node 侧）
// 职责：按 client 半的 wake/sleep 信号拉起/停止阿七 TTS 服务进程（:7890）。
// 路由：
//   POST /aqi-tts-api/wake   -> 确保服务在跑（开关开）
//   POST /aqi-tts-api/sleep  -> 停止服务（开关关，进程退出，显存/内存全归零）
//   GET  /aqi-tts-api/status -> { wanted, alive }
// 自愈：wanted 且进程死 → 30s 内重新 spawn；bind 冲突时新进程自动退出（旧实例继续服务）。
import { spawn } from 'node:child_process';
import http from 'node:http';

const HEALTH_URL = process.env.AQI_TTS_HEALTH || 'http://127.0.0.1:7890/health';

export const name = 'aqi-tts-service';
export const inject = ['webServer'];

// 可按机器修改（环境变量覆盖，便于开源后他人复用）
const PYTHON = process.env.AQI_TTS_PYTHON
  || 'F:\\Reasonix_work\\myslf\\_aqi\\py-faster\\python.exe';
const SCRIPT = process.env.AQI_TTS_SCRIPT
  || 'F:\\Reasonix_work\\myslf\\tools\\qwen_aqi_server.py';

export function apply(ctx) {
  const webServer = typeof ctx.get === 'function' ? ctx.get('webServer') : null;
  if (!webServer) return;

  let child = null;
  let wanted = false;

  const isAlive = () => child !== null && child.exitCode === null && !child.killed;

  // 端口上已有活实例（外部手动启动/上次遗留）→ 不重复 spawn，避免 spawn 风暴
  const portAlive = () => new Promise((resolve) => {
    const req = http.get(HEALTH_URL, { timeout: 1500 }, (res) => {
      res.resume();
      resolve(true);
    });
    req.on('error', () => resolve(false));
    req.on('timeout', () => { req.destroy(); resolve(false); });
  });

  const spawnServer = async () => {
    if (isAlive()) return;
    if (await portAlive()) return;
    try {
      child = spawn(PYTHON, [SCRIPT, '--preload'], {
        stdio: 'ignore',
        windowsHide: true,
      });
    } catch (e) {
      console.error('[aqi-tts-service] spawn failed:', (e && e.message) || e);
      return;
    }
    child.on('error', (e) => {
      console.error('[aqi-tts-service] child error:', (e && e.message) || e);
    });
    child.on('exit', (code) => {
      if (wanted && code !== 0 && code !== null) {
        // 端口已被旧实例占用而退出 = 正常自愈，旧实例仍在服务
        console.log(`[aqi-tts-service] python exited code=${code}`);
      }
    });
    console.log('[aqi-tts-service] python TTS server spawned');
  };

  const killServer = () => {
    if (!child) return;
    try { child.kill(); } catch { /* already gone */ }
    child = null;
    console.log('[aqi-tts-service] python TTS server stopped');
  };

  const json = (res, obj) => {
    const body = JSON.stringify(obj);
    res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
    res.end(body);
  };

  const disposers = [];
  disposers.push(webServer.register({
    kind: 'exact',
    path: '/aqi-tts-api/wake',
    async handler(req, res) {
      // 消耗请求体，避免连接报错
      for await (const _c of req) { /* drain */ }
      wanted = true;
      spawnServer();
      json(res, { ok: true, wanted, alive: isAlive() });
    },
  }));
  disposers.push(webServer.register({
    kind: 'exact',
    path: '/aqi-tts-api/sleep',
    async handler(req, res) {
      for await (const _c of req) { /* drain */ }
      wanted = false;
      killServer();
      json(res, { ok: true, wanted, alive: isAlive() });
    },
  }));
  disposers.push(webServer.register({
    kind: 'exact',
    path: '/aqi-tts-api/status',
    async handler(req, res) {
      json(res, { wanted, alive: isAlive() });
    },
  }));

  // 自愈循环：wanted 且进程不在 → 重拉
  const timer = setInterval(() => {
    if (wanted && !isAlive()) spawnServer();
  }, 30000);

  // DSH 退出 / 插件卸载：停掉 python，不留孤儿
  return () => {
    clearInterval(timer);
    killServer();
    for (const d of disposers) {
      try { if (typeof d === 'function') d(); } catch { /* ignore */ }
    }
  };
}
