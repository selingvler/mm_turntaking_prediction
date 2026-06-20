#!/usr/bin/env python3
"""
Browser GUI for the live LLM-gate service.

Two panels — USER metrics (camera / microphone / speaking enum / audio / transcript)
and SERVICE metrics (orchestrator state enum / shift score / LLM status / reply) —
plus a full, time-stamped EVENT timeline (every FIRE with its TRIGGER and the
message sent, LLM_READY with latency, SPEAK, DISCARD, barge-in…).

Open the printed http://localhost:PORT in a browser. The page polls /state.

API (call from the service loop, thread-safe):
    gui = WebGUI(); gui.start()
    gui.set_user(camera_fps=…, mic_rms=…, speaking=…, …)
    gui.set_service(state=…, score=…, thr=…, llm=…, …)
    gui.event(t, "FIRE_LLM", "trigger=speculative · sent: audio(2.3s)", kind="fire")
"""
from __future__ import annotations
import json
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


_HTML = """<!doctype html><html><head><meta charset=utf-8>
<title>MM-VAP LLM Gate</title><style>
 body{background:#0d1117;color:#e6edf3;font:14px/1.5 ui-monospace,Menlo,monospace;margin:0;padding:16px}
 h1{font-size:16px;margin:0 0 12px;color:#8b949e;font-weight:600}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
 .panel{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px}
 .panel h2{margin:0 0 10px;font-size:13px;letter-spacing:.08em;color:#58a6ff;text-transform:uppercase}
 .row{display:flex;justify-content:space-between;align-items:center;padding:5px 0;border-bottom:1px solid #21262d}
 .k{color:#8b949e}.v{font-weight:600}
 .bar{height:10px;background:#21262d;border-radius:6px;overflow:hidden;width:170px;display:inline-block;vertical-align:middle}
 .bar>i{display:block;height:100%}
 .badge{padding:2px 9px;border-radius:20px;font-weight:700;font-size:12px}
 .events{margin-top:14px;background:#161b22;border:1px solid #30363d;border-radius:10px;padding:12px;max-height:46vh;overflow:auto}
 .ev{display:flex;gap:10px;padding:3px 0;border-bottom:1px solid #21262d}
 .ev .t{color:#6e7681;min-width:74px}.ev .l{font-weight:700;min-width:150px}.ev .d{color:#adbac7}
 .fire{color:#d29922}.ready{color:#58a6ff}.speak{color:#3fb950}.bad{color:#f85149}.info{color:#8b949e}
 .ehead{display:flex;align-items:center;justify-content:space-between;margin:0 0 8px}
 button{background:#21262d;color:#e6edf3;border:1px solid #30363d;border-radius:6px;padding:5px 12px;font:inherit;cursor:pointer}
 button:hover{background:#30363d}
</style></head><body>
<h1>MM-VAP Turn-Taking · LLM Gate</h1>
<div class=grid>
 <div class=panel><h2>👤 User</h2><div id=user></div></div>
 <div class=panel><h2>⚙️ Service</h2><div id=service></div></div>
</div>
<div class=events>
 <div class=ehead>
   <h2 style="color:#58a6ff;font-size:13px;margin:0">⏱ Events</h2>
   <button onclick="copyEvents()" id=copybtn>📋 Copy events</button>
 </div>
 <div id=events></div></div>
<script>
const ST={LISTENING:'#1f6feb',SPECULATING:'#d29922',GRACE_WAIT:'#a371f7',SPEAKING:'#3fb950',COOLDOWN:'#6e7681'};
let LAST_EVENTS=[];
function copyEvents(){
 const lines=[...LAST_EVENTS].reverse().map(e=>`t=${e.t.toFixed(1)}s  ${e.label}  ${e.detail||''}`.trimEnd());
 const txt=lines.join('\\n')||'(no events)';
 navigator.clipboard.writeText(txt).then(()=>{
   const b=document.getElementById('copybtn');const o=b.textContent;
   b.textContent='✓ Copied!';setTimeout(()=>b.textContent=o,1200);
 }).catch(()=>{
   const ta=document.createElement('textarea');ta.value=txt;document.body.appendChild(ta);
   ta.select();document.execCommand('copy');ta.remove();
 });
}
function bar(frac,col){frac=Math.max(0,Math.min(1,frac));return `<span class=bar><i style="width:${frac*100}%;background:${col}"></i></span>`}
function row(k,v){return `<div class=row><span class=k>${k}</span><span class=v>${v}</span></div>`}
async function tick(){
 let s; try{s=await (await fetch('/state')).json()}catch(e){return}
 const u=s.user||{}, sv=s.service||{};
 let U='';
 U+=row('Camera', `${(u.camera_fps||0).toFixed(1)} fps`);
 U+=row('Microphone', bar((u.mic_rms||0)/((u.mic_gate||0.02)*4), (u.mic_rms>=u.mic_gate)?'#3fb950':'#6e7681')+` ${(u.mic_rms||0).toFixed(4)}`);
 U+=row('Speaking', u.speaking?'<span class=badge style="background:#3fb950;color:#04260f">● SPEAKING</span>':'<span class=badge style="background:#21262d;color:#8b949e">○ silent</span>');
 U+=row('Audio buffered', `${(u.audio_s||0).toFixed(1)} s`);
 U+=row('Heard (STT)', `<span class=v>${u.transcript||'—'}</span>`);
 document.getElementById('user').innerHTML=U;
 let V='';
 const col=ST[sv.state]||'#8b949e';
 V+=row('State', `<span class=badge style="background:${col};color:#fff">${sv.state||'—'}</span>`);
 V+=row(`Shift ${sv.signal||'p_now'}[robot]`, bar((sv.score||0)/((sv.thr||0.015)+(sv.softmax||2)), (sv.score>sv.thr)?'#3fb950':'#1f6feb')+` ${(sv.score||0).toFixed(3)} / thr ${(sv.thr||0).toFixed(3)}`);
 V+=row(`${sv.signal||'p_now'}[you] (hold)`, bar((sv.score_user||0)/((sv.softmax||2)),'#6e7681')+` ${(sv.score_user||0).toFixed(3)}`);
 V+=row('LLM', sv.llm||'idle');
 V+=row('Last reply', `<span class=v>${sv.last_reply||'—'}</span>`);
 V+=row('Perf', `cam ${(u.camera_fps||0).toFixed(0)}fps · predict ${(sv.predict_ms||0).toFixed(0)}ms · loop ${(sv.hop_hz||0).toFixed(1)}Hz`);
 document.getElementById('service').innerHTML=V;
 LAST_EVENTS=s.events||[];
 let E='';
 for(const e of (s.events||[])){
   E+=`<div class=ev><span class=t>t=${e.t.toFixed(1)}s</span><span class="l ${e.kind||'info'}">${e.label}</span><span class=d>${e.detail||''}</span></div>`}
 document.getElementById('events').innerHTML=E;
}
setInterval(tick,150);tick();
</script></body></html>"""


class WebGUI:
    def __init__(self, port: int = 8765, max_events: int = 200):
        self.port = port
        self._lock = threading.Lock()
        self._user: dict = {}
        self._service: dict = {}
        self._events: deque = deque(maxlen=max_events)
        self._srv = None

    # ── service-side API (thread-safe) ───────────────────────────────
    def set_user(self, **kw):
        with self._lock:
            self._user.update(kw)

    def set_service(self, **kw):
        with self._lock:
            self._service.update(kw)

    def event(self, t: float, label: str, detail: str = "", kind: str = "info"):
        with self._lock:
            self._events.appendleft({"t": float(t), "label": label,
                                     "detail": detail, "kind": kind})

    def _snapshot(self) -> bytes:
        with self._lock:
            return json.dumps({"user": self._user, "service": self._service,
                               "events": list(self._events)}).encode()

    # ── server ───────────────────────────────────────────────────────
    def start(self):
        gui = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence access logs
                pass

            def do_GET(self):
                if self.path.startswith("/state"):
                    body = gui._snapshot(); ctype = "application/json"
                else:
                    body = _HTML.encode(); ctype = "text/html; charset=utf-8"
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._srv = ThreadingHTTPServer(("127.0.0.1", self.port), H)
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()
        print(f"GUI:  http://localhost:{self.port}   (open in a browser)")
        return self

    def stop(self):
        if self._srv:
            self._srv.shutdown()


if __name__ == "__main__":
    import time, math
    g = WebGUI().start()
    g.event(0.0, "FIRE_LLM", "trigger=speculative · sent: audio(2.3s)", "fire")
    g.event(1.2, "LLM_READY", "latency=1.20s", "ready")
    g.event(1.5, "SPEAK", "İyiyim, teşekkürler.", "speak")
    i = 0
    while True:
        i += 1
        g.set_user(camera_fps=27, mic_rms=0.02 + 0.15*abs(math.sin(i/6)),
                   mic_gate=0.012, speaking=(i // 6) % 2 == 0, audio_s=2.1,
                   transcript="merhaba nasılsın")
        g.set_service(state=["LISTENING", "SPECULATING", "SPEAKING"][i % 3],
                      score=abs(math.sin(i/5)), thr=0.015,
                      llm="⏳ waiting 0.4s", last_reply="İyiyim.",
                      predict_ms=140, hop_hz=1.9)
        time.sleep(0.3)
