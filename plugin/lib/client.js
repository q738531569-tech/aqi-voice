// aqi-tts-service —— Client 半（浏览器侧）
// 监听 dsh-plugin-tts 的「自动朗读」开关（localStorage["dsh-tts-settings"].autoRead），
// 开 -> POST /aqi-tts-api/wake（拉起服务）；关 -> POST /aqi-tts-api/sleep（停止服务）。
// 另每 30s 补发一次 wake 心跳（防 Host 重启丢状态；Host 侧幂等，重复 wake 无害）。
window.__ModuleLoader__.load({
  id: "aqi-tts-service",
  factory: function (require) {
    var module = { exports: {} };
    var exports = module.exports;
    Object.defineProperty(exports, Symbol.toStringTag, { value: "Module" });

    const inject = [];

    const SETTINGS_KEY = "dsh-tts-settings"; // dsh-plugin-tts 的共享设置（含 autoRead）
    const CLIENT_BUILD = "aqi-tts-0.2.0";

    const apply = function (ctx) {
      let last = null; // null=未知，true/false=已发送过的开关态

      const readSwitch = function () {
        try {
          const raw = localStorage.getItem(SETTINGS_KEY);
          if (!raw) return false;
          return !!(JSON.parse(raw).autoRead);
        } catch (e) {
          return false;
        }
      };

      const ping = function (route) {
        try {
          fetch("/aqi-tts-api/" + route, { method: "POST" }).catch(function () { /* Host 未就绪，下轮重试 */ });
        } catch (e) { /* ignore */ }
      };

      let n = 0;
      const tick = function () {
        const on = readSwitch();
        if (on !== last) {
          last = on;
          ping(on ? "wake" : "sleep");
        } else if (on && ++n % 15 === 0) {
          ping("wake"); // 2s*15=30s 心跳：Host 重启后由 client 重新拉起（幂等）
        }
      };

      tick();
      const timer = setInterval(tick, 2000);
      if (typeof ctx.on === "function") {
        // 页面卸载时不刻意 sleep：DSH 退出会由 Host 半负责杀进程
      }
      return function () { clearInterval(timer); };
    };

    exports.inject = inject;
    exports.apply = apply;
    return module.exports;
  },
});
