#!/usr/bin/env python3
"""
Subtitle Speech Universal WebPlayer - Fast Transcript Engine

Backend order:
  1) Groq Whisper (when GROQ_API_KEY is configured) - fastest practical cloud path
  2) whisper.cpp CLI (when WHISPER_CPP_BIN + WHISPER_CPP_MODEL are configured)
  3) faster-whisper local CPU INT8

The engine is deliberately isolated from the extension UI, subtitle translation,
and TTS logic. It returns the same segment shape used by bridge_server.py.
"""
from __future__ import annotations
import json
import os
import re
import subprocess
import tempfile
import threading
from typing import Any

_lock = threading.Lock()
_fw_model = None


def _norm_lang(language: str | None) -> str | None:
    x = (language or os.environ.get("WHISPER_LANG", "") or "").strip().lower()
    if not x:
        return None
    if x in {"chinese", "cn", "zh-cn", "zh-tw", "zh-hans", "zh-hant"}:
        return "zh"
    return x


def _clean(text: Any) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text


def _fw():
    global _fw_model
    from faster_whisper import WhisperModel
    if _fw_model is None:
        with _lock:
            if _fw_model is None:
                model_name = os.environ.get("WHISPER_MODEL", "tiny")
                device = os.environ.get("WHISPER_DEVICE", "cpu")
                compute = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
                print(f"[fast-stt] loading faster-whisper model={model_name} device={device} compute={compute}")
                _fw_model = WhisperModel(model_name, device=device, compute_type=compute)
    return _fw_model


def transcribe_faster_whisper(wav: str, language: str = ""):
    model = _fw()
    lang = _norm_lang(language)
    beam = max(1, int(os.environ.get("FAST_WHISPER_BEAM", "1")))
    best_of = max(1, int(os.environ.get("FAST_WHISPER_BEST_OF", "1")))
    with _lock:
        segments, info = model.transcribe(
            wav,
            beam_size=beam,
            best_of=best_of,
            language=lang,
            task="transcribe",
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": int(os.environ.get("FAST_VAD_SILENCE_MS", "350")),
                "speech_pad_ms": int(os.environ.get("FAST_VAD_PAD_MS", "250")),
                "threshold": float(os.environ.get("FAST_VAD_THRESHOLD", "0.35")),
            },
            condition_on_previous_text=False,
            without_timestamps=False,
            temperature=0.0,
            no_speech_threshold=0.45,
            compression_ratio_threshold=2.8,
        )
    out = []
    for seg in segments:
        text = _clean(seg.text)
        if text:
            out.append({"start": float(seg.start), "end": float(seg.end), "text": text})
    return out, getattr(info, "language", "") or "", "faster-whisper-fast"


def _parse_whisper_cpp_json(path: str):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)
    candidates = data.get("transcription") or data.get("segments") or []
    out = []
    for item in candidates:
        text = _clean(item.get("text"))
        if not text:
            continue
        # whisper.cpp JSON normally stores offsets in milliseconds.
        off = item.get("offsets") or {}
        start = item.get("start", off.get("from", 0))
        end = item.get("end", off.get("to", start))
        start = float(start or 0)
        end = float(end or start)
        if start > 1000 or end > 1000:
            start /= 1000.0
            end /= 1000.0
        out.append({"start": start, "end": max(start, end), "text": text})
    return out


def transcribe_whisper_cpp(wav: str, language: str = ""):
    binary = os.environ.get("WHISPER_CPP_BIN", "").strip()
    model = os.environ.get("WHISPER_CPP_MODEL", "").strip()
    if not binary or not model:
        raise RuntimeError("whisper.cpp backend not configured")
    if not os.path.isfile(binary):
        raise RuntimeError(f"whisper.cpp binary not found: {binary}")
    if not os.path.isfile(model):
        raise RuntimeError(f"whisper.cpp model not found: {model}")
    with tempfile.TemporaryDirectory(prefix="uss-wcpp-") as td:
        outbase = os.path.join(td, "result")
        cmd = [
            binary, "-m", model, "-f", wav,
            "-oj", "-of", outbase,
            "-bs", os.environ.get("WHISPER_CPP_BEAM", "1"),
            "-bo", os.environ.get("WHISPER_CPP_BEST_OF", "1"),
            "-np",
        ]
        lang = _norm_lang(language)
        if lang:
            cmd += ["-l", lang]
        else:
            cmd += ["-l", "auto"]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=int(os.environ.get("WHISPER_CPP_TIMEOUT", "1800")))
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or b"").decode("utf-8", "ignore")[-800:]
            raise RuntimeError("whisper.cpp failed: " + err)
        json_path = outbase + ".json"
        if not os.path.isfile(json_path):
            raise RuntimeError("whisper.cpp produced no JSON transcript")
        return _parse_whisper_cpp_json(json_path), lang or "", "whisper.cpp-fast"


def _find_cpp_binary():
    explicit = os.environ.get("WHISPER_CPP_BIN", "").strip()
    candidates = [explicit] if explicit else []
    prefix = os.environ.get("PREFIX", "").strip()
    home = os.path.expanduser("~")
    candidates += [
        os.path.join(prefix, "bin", "whisper-cli") if prefix else "",
        os.path.join(prefix, "bin", "main") if prefix else "",
        os.path.join(home, "whisper.cpp", "build", "bin", "whisper-cli"),
        os.path.join(home, "whisper.cpp", "build", "bin", "main"),
        "/usr/local/bin/whisper-cli",
        "/usr/bin/whisper-cli",
    ]
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return ""

def _find_cpp_model():
    explicit = os.environ.get("WHISPER_CPP_MODEL", "").strip()
    candidates = [explicit] if explicit else []
    home = os.path.expanduser("~")
    candidates += [
        os.path.join(home, "whisper.cpp", "models", "ggml-tiny.bin"),
        os.path.join(home, "whisper.cpp", "models", "ggml-base.bin"),
        os.path.join(home, ".cache", "whisper.cpp", "ggml-tiny.bin"),
        "/data/data/com.termux/files/home/whisper.cpp/models/ggml-tiny.bin",
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return ""

def transcribe_wav(wav: str, language: str = ""):
    """Fast local backend. Never calls a cloud transcription service."""
    errors = []
    cpp_bin = _find_cpp_binary()
    cpp_model = _find_cpp_model()
    if cpp_bin and cpp_model:
        try:
            os.environ["WHISPER_CPP_BIN"] = cpp_bin
            os.environ["WHISPER_CPP_MODEL"] = cpp_model
            return transcribe_whisper_cpp(wav, language)
        except Exception as exc:
            errors.append("whisper.cpp: " + str(exc))
            print("[fast-stt] whisper.cpp failed:", exc)
    try:
        return transcribe_faster_whisper(wav, language)
    except Exception as exc:
        errors.append("faster-whisper: " + str(exc))
    raise RuntimeError("Fast Offline Transcript Engine has no working local backend. " + " | ".join(errors))
