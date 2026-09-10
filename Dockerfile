FROM python:3.11-slim

# ffmpeg is required by faster-whisper / yt-dlp for audio decoding.
# curl + unzip are needed to install Deno.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl unzip \
    && rm -rf /var/lib/apt/lists/*

# yt-dlp needs a JS runtime (Deno) to solve YouTube's playback challenge.
ENV DENO_INSTALL="/usr/local"
RUN curl -fsSL https://deno.land/install.sh | sh
ENV PATH="/usr/local/bin:${PATH}"

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    # YouTube's anti-bot checks change often — always use the newest yt-dlp,
    # not whatever version requirements.txt happens to pin.
    && pip install --no-cache-dir -U yt-dlp

COPY . .

# Hugging Face Spaces expects the app to listen on port 7860
ENV PORT=7860
EXPOSE 7860

CMD ["python", "bridge_server.py"]
