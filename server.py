"""Local-only reading practice server, no audio upload to cloud.

Python 3.10+. Model downloaded once on first transcription, then cached locally.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
HOST = "127.0.0.1"
PORT = int(os.environ.get("READING_BUDDY_PORT", "8765"))
MAX_AUDIO_BYTES = 3 * 1024 * 1024
MAX_AUDIO_SECONDS = 12
_MODEL = None
_MODEL_LOCK = threading.Lock()


def normalize(text: str) -> str:
    """Normalize recognition only for word/phrase string equality, not phoneme scoring."""
    text = text.casefold().replace("’", "'")
    text = re.sub(r"[^a-z0-9'\s-]", " ", text)
    return " ".join(text.replace("-", " ").split())


def get_model():
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            from faster_whisper import WhisperModel
            _MODEL = WhisperModel("tiny.en", device="cpu", compute_type="int8")
        return _MODEL


def transcribe_audio(audio: bytes, suffix: str) -> str:
    # File is deleted even on error; never keep or log the child's voice.
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio)
            tmp_path = f.name
        # This tiny model may mishear isolated words. Treat recognition as a hint only.
        model = get_model()
        # Without a target-word hint, recognition is less likely to parrot the prompt.
        with _MODEL_LOCK:
            segments, _info = model.transcribe(
                tmp_path, language="en", beam_size=1,
                condition_on_previous_text=False,
                vad_filter=False,
            )
            return " ".join(s.text.strip() for s in segments).strip()
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # Do not log recording, recognized content, or request body.
        if self.path.startswith("/api/"):
            return
        super().log_message(fmt, *args)

    def reply(self, status: int, payload: dict):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/api/health":
            try:
                import faster_whisper  # noqa: F401
                installed = True
            except ImportError:
                installed = False
            return self.reply(200, {
                "mode": "local", "aiInstalled": installed,
                "modelLoaded": _MODEL is not None,
                "message": "语音仅在本机处理，不上传云端",
            })
        mapping = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/index.html": ("index.html", "text/html; charset=utf-8"),
            "/style.css": ("style.css", "text/css; charset=utf-8"),
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
        }
        if path not in mapping:
            return self.send_error(404)
        name, media_type = mapping[path]
        data = (STATIC / name).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", media_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; media-src 'self' blob:; style-src 'self'; script-src 'self'; connect-src 'self'; base-uri 'none'; object-src 'none'")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if urlsplit(self.path).path != "/api/recognize":
            return self.reply(404, {"error": "未找到接口"})
        origin = self.headers.get("Origin", "")
        allowed = {f"http://localhost:{PORT}", f"http://127.0.0.1:{PORT}"}
        if origin not in allowed or self.headers.get("X-Reading-Buddy") != "local-only":
            return self.reply(403, {"error": "只允许本机页面发起识别"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if not 300 <= length <= MAX_AUDIO_BYTES:
            return self.reply(413, {"error": "录音太短或文件过大，请重试"})
        content_type = self.headers.get("Content-Type", "").split(";")[0].lower()
        suffixes = {
            "audio/webm": ".webm", "audio/mp4": ".mp4",
            "audio/ogg": ".ogg", "audio/wav": ".wav",
            "audio/x-wav": ".wav", "audio/mpeg": ".mp3",
        }
        if content_type not in suffixes:
            return self.reply(415, {"error": "浏览器的录音格式暂不支持，请使用新版 Chrome 或 Edge"})
        content = self.rfile.read(length)
        try:
            transcript = transcribe_audio(content, suffixes[content_type])
        except ModuleNotFoundError:
            return self.reply(503, {"error": "本地 AI 未安装，请先运行 pip install -r requirements.txt"})
        except Exception:
            return self.reply(503, {"error": "AI 识别暂时失败。请检查网络是否能下载首次使用的模型，或稍后重试"})
        return self.reply(200, {"heard": transcript})


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"\n🎀 每日朗读小老师已经启动：http://localhost:{PORT}\n")
    print("模型首次使用会下载 tiny.en，之后可离线识别。按 Ctrl+C 停止。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
