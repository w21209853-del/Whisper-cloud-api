#!/usr/bin/env python3
"""
SubtitleSpeaker Python bridge — stable fetch with retries.

Run in Termux:
  pip install -U youtube-transcript-api
  python bridge_server.py

If you see DNS / NameResolutionError:
  - Check internet:  ping -c 2 8.8.8.8
  - Check DNS:      ping -c 2 www.youtube.com
  - Try:            setprop net.dns1 8.8.8.8   (needs root)
                    or use private DNS 8.8.8.8 / 1.1.1.1 in Android settings
"""
from __future__ import annotations

import http.server
import base64
import subprocess
import tempfile
import json
import re
import socket
import socketserver
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict

try:
    from youtube_transcript_api import YouTubeTranscriptApi
except ImportError:
    print("ERROR: pip install -U youtube-transcript-api")
    raise

# Local Termux default. On Render/Railway/etc: HOST=0.0.0.0 and PORT from env.
import os
HOST = os.environ.get("HOST", "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1")
PORT = int(os.environ.get("PORT", "8765"))

# Brief in-memory cache so Collect twice does not re-hit YouTube every time.
_CACHE_MAX = 8
_CACHE_TTL_SEC = 600
_cache: OrderedDict = OrderedDict()

MAX_RETRIES = 4
RETRY_BASE_SEC = 1.2

# Optional local speech-to-text layer used only by the Transcript feature.
# It is lazy-loaded so the existing subtitle bridge starts even when Whisper
# dependencies are not installed yet.
_WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "tiny")
_whisper = None
_whisper_lock = __import__("threading").Lock()

def transcribe_audio_bytes(raw: bytes, mime: str = "audio/webm", language: str = ""):
    """Fast Transcript Engine entrypoint for short captured audio."""
    suffix = ".ogg" if "ogg" in mime else ".webm"
    with tempfile.TemporaryDirectory(prefix="uss-stt-") as td:
        src = os.path.join(td, "chunk" + suffix)
        wav = os.path.join(td, "audio.wav")
        with open(src, "wb") as f:
            f.write(raw)
        try:
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src,
                            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav],
                           check=True, timeout=45, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg is not installed") from exc
        except subprocess.CalledProcessError as exc:
            err = exc.stderr.decode("utf-8", "ignore")[-500:] if exc.stderr else "ffmpeg failed"
            raise RuntimeError(err) from exc
        from fast_transcript_engine import transcribe_wav
        segments, detected, engine = transcribe_wav(wav, language)
        return {"ok": True, "segments": segments, "segmentsCount": len(segments),
                "language": detected or language or "", "engine": engine}


def _segments_from_whisper_wav(wav: str, language: str = ""):
    """Fast Transcript Engine full-file path."""
    from fast_transcript_engine import transcribe_wav
    segments, detected, engine = transcribe_wav(wav, language)
    return segments, detected, engine


def transcribe_via_groq(raw: bytes, filename: str = "audio.wav", language: str = "", source_path: str = ""):
    """Optional fast cloud path — set GROQ_API_KEY in the environment.
    Compresses large WAV to small MP3 first to avoid Broken pipe on mobile networks.
    """
    key = (os.environ.get("GROQ_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY not set")

    # Prefer compact MP3 upload (Groq accepts; much smaller than WAV → fewer Broken pipe)
    upload_data = raw
    upload_name = filename or "audio.wav"
    try:
        src = (source_path or "").strip()
        if src and os.path.isfile(src):
            mp3 = src + ".groq.mp3"
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", src,
                    "-vn", "-ac", "1", "-ar", "16000",
                    "-c:a", "libmp3lame", "-b:a", "48k",
                    mp3,
                ],
                check=True,
                timeout=300,
            )
            with open(mp3, "rb") as f:
                upload_data = f.read()
            upload_name = "audio.mp3"
            try:
                os.remove(mp3)
            except Exception:
                pass
        elif len(raw) > 3_000_000:
            # bytes only: write temp wav then compress
            with tempfile.TemporaryDirectory(prefix="uss-groq-") as td:
                wav_p = os.path.join(td, "in.wav")
                mp3_p = os.path.join(td, "out.mp3")
                with open(wav_p, "wb") as f:
                    f.write(raw)
                subprocess.run(
                    [
                        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-i", wav_p,
                        "-vn", "-ac", "1", "-ar", "16000",
                        "-c:a", "libmp3lame", "-b:a", "48k",
                        mp3_p,
                    ],
                    check=True,
                    timeout=300,
                )
                with open(mp3_p, "rb") as f:
                    upload_data = f.read()
                upload_name = "audio.mp3"
    except Exception as exc:
        print("[stt] Groq compress skipped:", exc)

    lang = (language or "").strip()
    if lang in ("chinese", "cn", "zh-cn", "zh-tw"):
        lang = "zh"

    def _one_post():
        import urllib.request as ur
        boundary = "----USS" + str(int(time.time() * 1000))
        parts = []
        def add_field(name, value):
            parts.append(
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()
            )
        def add_file(name, fname, data):
            parts.append(
                (
                    f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; "
                    f"filename=\"{fname}\"\r\nContent-Type: application/octet-stream\r\n\r\n"
                ).encode()
                + data
                + b"\r\n"
            )
        add_file("file", upload_name, upload_data)
        add_field("model", os.environ.get("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo"))
        add_field("response_format", "verbose_json")
        if lang:
            add_field("language", lang)
        body = b"".join(parts) + f"--{boundary}--\r\n".encode()
        req = ur.Request(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            data=body,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(body)),
            },
            method="POST",
        )
        with ur.urlopen(req, timeout=600) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))

    last_err = None
    data = None
    for attempt in range(1, 4):
        try:
            print(f"[stt] Groq upload attempt {attempt} size={len(upload_data)} name={upload_name}")
            data = _one_post()
            break
        except Exception as exc:
            last_err = exc
            err_s = str(exc)
            print(f"[stt] Groq attempt {attempt} failed:", exc)
            # 403 = bad/blocked API key — retrying is useless, go local immediately
            if "403" in err_s or "Forbidden" in err_s:
                print("[stt] Groq 403 Forbidden — API key rejected. Skipping retries, using local Whisper.")
                raise RuntimeError("Groq 403 Forbidden (check API key at console.groq.com)") from exc
            time.sleep(1.5 * attempt)
    if data is None:
        raise RuntimeError(str(last_err or "Groq failed"))

    out = []
    for seg in data.get("segments") or []:
        text = str(seg.get("text") or "").strip()
        if text:
            out.append({
                "start": float(seg.get("start") or 0),
                "end": float(seg.get("end") or 0),
                "text": text,
            })
    if not out and data.get("text"):
        out.append({"start": 0.0, "end": 0.0, "text": str(data.get("text") or "").strip()})
    return {
        "ok": True,
        "segments": out,
        "segmentsCount": len(out),
        "language": language or str(data.get("language") or ""),
        "engine": "groq",
    }



def transcribe_media_url(url: str, language: str = ""):
    """Full offline transcript: download/extract audio, then transcribe locally only."""
    url = (url or "").strip()
    if not url.startswith("http"):
        raise RuntimeError("invalid media url")
    with tempfile.TemporaryDirectory(prefix="uss-full-") as td:
        outtmpl = os.path.join(td, "media.%(ext)s")
        cmd = [
            "yt-dlp", "-f", "bestaudio/best", "-x", "--audio-format", "wav",
            "--audio-quality", "0", "-o", outtmpl, "--no-playlist", "--newline",
            "--user-agent",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
            "--add-header", "Accept-Language:en-US,en;q=0.9", "--retries", "5",
            "--fragment-retries", "5", "--retry-sleep", "1", url,
        ]
        cookies = (os.environ.get("YTDLP_COOKIES") or "").strip()
        if cookies and os.path.isfile(cookies):
            cmd[1:1] = ["--cookies", cookies]
        try:
            proc = subprocess.run(cmd, check=False, timeout=int(os.environ.get("YTDLP_TIMEOUT", "3600")),
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError as exc:
            raise RuntimeError("yt-dlp not installed. Install: pip install -U yt-dlp") from exc
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or b"").decode("utf-8", "ignore")[-1200:]
            raise RuntimeError("yt-dlp failed: " + err)
        wav = None
        for name in os.listdir(td):
            if name.lower().endswith((".wav", ".mp3", ".m4a", ".webm", ".ogg", ".opus", ".mp4")):
                wav = os.path.join(td, name)
                break
        if not wav:
            raise RuntimeError("yt-dlp produced no audio file")
        if not wav.lower().endswith(".wav"):
            wav2 = os.path.join(td, "audio16k.wav")
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", wav,
                            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav2],
                           check=True, timeout=600)
            wav = wav2
        print("[stt] OFFLINE full media ready bytes=", os.path.getsize(wav))
        print("[stt] OFFLINE local engine start…")
        segs, lang, engine = _segments_from_whisper_wav(wav, language)
        print("[stt] OFFLINE local engine done segments=", len(segs), "engine=", engine)
        return {
            "ok": True, "segments": segs, "segmentsCount": len(segs),
            "language": lang or language or "", "engine": engine, "offline": True,
        }


def extract_video_id(url: str):
    patterns = [
        r"(?:v=)([0-9A-Za-z_-]{11})",
        r"(?:youtu\.be/)([0-9A-Za-z_-]{11})",
        r"(?:youtube\.com/embed/)([0-9A-Za-z_-]{11})",
        r"(?:youtube\.com/shorts/)([0-9A-Za-z_-]{11})",
        r"^([0-9A-Za-z_-]{11})$",
    ]
    for pattern in patterns:
        match = re.search(pattern, url or "")
        if match:
            return match.group(1)
    return None


def snippet_value(item, name, default=None):
    value = getattr(item, name, None)
    if value is not None:
        return value
    if isinstance(item, dict):
        return item.get(name, default)
    return default


def cache_get(video_id: str):
    entry = _cache.get(video_id)
    if not entry:
        return None
    data, ts = entry
    if time.time() - ts > _CACHE_TTL_SEC:
        _cache.pop(video_id, None)
        return None
    # LRU touch
    _cache.move_to_end(video_id)
    return data


def cache_set(video_id: str, data):
    _cache[video_id] = (data, time.time())
    _cache.move_to_end(video_id)
    while len(_cache) > _CACHE_MAX:
        _cache.popitem(last=False)


def is_network_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    needles = (
        "nameresolutionerror",
        "failed to resolve",
        "no address associated",
        "nodename nor servname",
        "temporary failure in name resolution",
        "max retries exceeded",
        "connection aborted",
        "connection reset",
        "timed out",
        "timeout",
        "network is unreachable",
        "errno 7",
        "errno 110",
        "errno 101",
        "ssl",
    )
    return any(n in msg for n in needles)


def friendly_error(exc: BaseException) -> str:
    msg = str(exc)
    low = msg.lower()

    if is_network_error(exc):
        return (
            "Network/DNS error: bridge could not reach YouTube. "
            "Check internet / Private DNS (dns.google). Detail: " + msg[:200]
        )

    if "private" in low:
        return (
            "এই ভিডিও Private — subtitle নেওয়া যায় না। "
            "Public ভিডিও খুলুন যেখানে CC/Captions আছে। | " + msg[:160]
        )

    if "unavailable" in low or "unplayable" in low:
        return (
            "ভিডিও play/unplayable (region/age/removed)। "
            "অন্য public ভিডিও try করুন যেখানে subtitle আছে। | " + msg[:160]
        )

    if "disabled" in low or "subtitles are disabled" in low:
        return (
            "এই ভিডিওতে subtitle বন্ধ করা আছে (uploader disabled captions)। "
            "CC আছে এমন ভিডিও দরকার। | " + msg[:140]
        )

    if "no transcript" in low or "could not retrieve a transcript" in low or \
       "no element found" in low or "no subtitle" in low:
        return (
            "এই ভিডিওতে কোনো transcript/CC নেই "
            "(music/instrumental বা captions off)। "
            "YouTube-এ CC বাটন আছে এমন ভিডিও খুলুন। | " + msg[:140]
        )

    if "age" in low and "restrict" in low:
        return (
            "Age-restricted ভিডিও — login ছাড়া transcript পাওয়া যায় না। "
            "অন্য ভিডিও try করুন।"
        )

    return msg[:400]


def check_dns(host: str = "www.youtube.com", timeout: float = 3.0) -> dict:
    try:
        infos = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        addrs = sorted({i[4][0] for i in infos})
        return {"ok": True, "host": host, "addresses": addrs[:6]}
    except Exception as exc:
        return {"ok": False, "host": host, "error": str(exc)}


def choose_transcript(video_id: str):
    api = YouTubeTranscriptApi()
    transcript_list = api.list(video_id)
    tracks = list(transcript_list)

    if not tracks:
        raise RuntimeError(
            "No subtitle tracks available for this video "
            "(no CC / captions on YouTube)"
        )

    # Prefer: en manual → en generated → bn → any manual → any generated → first
    def lang(t):
        return (getattr(t, "language_code", "") or "").lower()

    def generated(t):
        return bool(getattr(t, "is_generated", False))

    ordered = sorted(
        tracks,
        key=lambda t: (
            0 if lang(t).startswith("en") and not generated(t) else
            1 if lang(t).startswith("en") else
            2 if lang(t).startswith("bn") else
            3 if not generated(t) else
            4
        )
    )
    return ordered[0]


def snippets_to_subtitles(fetched):
    subtitles = []
    for item in fetched:
        start = float(snippet_value(item, "start", 0) or 0)
        duration = float(snippet_value(item, "duration", 0) or 0)
        text = str(snippet_value(item, "text", "") or "").replace("\n", " ").strip()
        if not text:
            continue
        subtitles.append({
            "no": len(subtitles) + 1,
            "time": f"{int(start // 60):02d}:{int(start % 60):02d}",
            "start": start,
            "duration": duration,
            "end": start + duration,
            "text": text,
            "translated": "",
        })
    return subtitles


def fetch_subtitles_once(video_id: str):
    """Try list()+fetch, then direct get_transcript style fallbacks."""
    last_err = None

    # Path A: list available tracks and fetch best one
    try:
        transcript = choose_transcript(video_id)
        fetched = transcript.fetch()
        subtitles = snippets_to_subtitles(fetched)
        if subtitles:
            return {
                "subtitles": subtitles,
                "language": getattr(transcript, "language_code", "") or "",
            }
        last_err = RuntimeError("Subtitle track empty")
    except Exception as exc:
        last_err = exc

    # Path B: try common languages via API.fetch if available
    try:
        api = YouTubeTranscriptApi()
        for langs in (["en"], ["en-US", "en-GB"], ["bn"], ["hi"], None):
            try:
                if hasattr(api, "fetch"):
                    if langs is None:
                        fetched = api.fetch(video_id)
                    else:
                        fetched = api.fetch(video_id, languages=langs)
                elif hasattr(YouTubeTranscriptApi, "get_transcript"):
                    # Older API
                    raw = YouTubeTranscriptApi.get_transcript(
                        video_id, languages=langs or ["en", "bn", "hi"]
                    )
                    fetched = raw
                else:
                    break
                subtitles = snippets_to_subtitles(fetched)
                if subtitles:
                    lang = (langs[0] if langs else "") or "en"
                    return {"subtitles": subtitles, "language": lang}
            except Exception as exc:
                last_err = exc
                continue
    except Exception as exc:
        last_err = exc

    if last_err is not None:
        raise last_err
    raise RuntimeError("No transcript available for this video")


def get_complete_subtitles(video_id: str):
    cached = cache_get(video_id)
    if cached:
        return cached["subtitles"], cached["language"], True

    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            data = fetch_subtitles_once(video_id)
            cache_set(video_id, data)
            return data["subtitles"], data["language"], False
        except Exception as exc:
            last_exc = exc
            if not is_network_error(exc) and attempt >= 2:
                # Non-network (e.g. no captions) — no point hammering.
                break
            if attempt < MAX_RETRIES:
                sleep_for = RETRY_BASE_SEC * (1.6 ** (attempt - 1))
                print(f"[bridge] attempt {attempt}/{MAX_RETRIES} failed: {exc}")
                print(f"[bridge] retry in {sleep_for:.1f}s…")
                time.sleep(sleep_for)

    assert last_exc is not None
    raise last_exc


class Handler(http.server.BaseHTTPRequestHandler):
    def _headers(self):
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._headers()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self._headers()
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/health":
            dns = check_dns()
            self.send_json({
                "ok": True,
                "service": "SubtitleSpeaker bridge",
                "version": "1.13.0-fast-engine",
                "dns": dns,
                "cache_size": len(_cache),
                "stt": "fast-transcript-engine",
                "whisper_model": _WHISPER_MODEL,
                "offline_transcript": True,
            })
            return

        if parsed.path != "/get_subtitles":
            self.send_json({"error": "Not found"}, 404)
            return

        query = urllib.parse.parse_qs(parsed.query)
        video_url = query.get("url", [""])[0]
        video_id = extract_video_id(video_url)

        if not video_id:
            self.send_json({"success": False, "error": "Invalid YouTube URL"}, 400)
            return

        try:
            subtitles, source_language, from_cache = get_complete_subtitles(video_id)
            self.send_json({
                "success": True,
                "video_id": video_id,
                "language": source_language,
                "count": len(subtitles),
                "cached": from_cache,
                "subtitles": subtitles,
            })
        except Exception as exc:
            err = friendly_error(exc)
            print("[bridge] ERROR:", err)
            traceback.print_exc()
            self.send_json({
                "success": False,
                "error": err,
                "video_id": video_id,
                "dns": check_dns(),
            }, 200)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path not in ("/transcribe_audio", "/transcribe_media_url"):
            self.send_json({"ok": False, "error": "Not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 80 * 1024 * 1024:
                raise RuntimeError("invalid audio payload")
            body = self.rfile.read(length)
            payload = json.loads(body.decode("utf-8"))
            if parsed.path == "/transcribe_media_url":
                media_url = str(payload.get("url") or payload.get("mediaUrl") or "").strip()
                result = transcribe_media_url(media_url, str(payload.get("language", "") or ""))
                result.update({"session": payload.get("session", ""), "mode": "full-url"})
                self.send_json(result)
                return
            audio = base64.b64decode(str(payload.get("audio", "")), validate=True)
            if not audio:
                raise RuntimeError("empty audio")
            result = transcribe_audio_bytes(audio, str(payload.get("mime", "audio/webm")), str(payload.get("language", "") or ""))
            result.update({"session": payload.get("session", ""), "seq": payload.get("seq", 0), "start": payload.get("start", 0), "end": payload.get("end", 0)})
            self.send_json(result)
        except Exception as exc:
            err = friendly_error(exc)
            print("[stt] ERROR:", err)
            self.send_json({"ok": False, "error": err}, 200)

    def log_message(self, format, *args):
        print("[bridge] " + (format % args))


class ReusableTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def server_bind(self):
        try:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except Exception:
            pass
        super().server_bind()


if __name__ == "__main__":
    print(f"🚀 SubtitleSpeaker Python bridge: http://{HOST}:{PORT}")
    print("   Keep this terminal running while using the extension.")
    dns = check_dns()
    if dns.get("ok"):
        print(f"   DNS OK → www.youtube.com = {', '.join(dns.get('addresses') or [])}")
    else:
        print("   ⚠️  DNS FAIL for www.youtube.com:", dns.get("error"))
        print("   Fix Private DNS (dns.google) or check mobile data/Wi‑Fi, then restart bridge.")
    with ReusableTCPServer((HOST, PORT), Handler) as server:
        server.serve_forever()
