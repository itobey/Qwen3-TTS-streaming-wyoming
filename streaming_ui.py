#!/usr/bin/env python3
"""Standalone FastAPI server with HTML UI for true streaming audio."""

import argparse
import asyncio
import base64
import json
import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

try:
    from wyoming.audio import AudioChunk, AudioStart, AudioStop
    from wyoming.event import async_read_event, async_write_event
    from wyoming.info import Describe
    from wyoming.tts import Synthesize
    WYOMING_AVAILABLE = True
except ImportError:
    WYOMING_AVAILABLE = False

_LOGGER = logging.getLogger(__name__)


async def wyoming_audio_generator(text: str, host: str, port: int):
    """Generate audio chunks from Wyoming server."""
    if not WYOMING_AVAILABLE:
        yield json.dumps({"type": "error", "message": "Wyoming not installed"}) + "\n"
        return
    
    reader = None
    writer = None
    
    try:
        _LOGGER.info(f"Connecting to Wyoming at {host}:{port}")
        yield json.dumps({"type": "status", "message": f"🔌 Connected to {host}:{port}"}) + "\n"
        
        reader, writer = await asyncio.open_connection(host, port)
        
        await async_write_event(Describe().event(), writer)
        await async_read_event(reader)
        
        _LOGGER.info(f"Synthesizing: {text[:50]}...")
        yield json.dumps({"type": "status", "message": f"📤 Synthesizing..."}) + "\n"
        
        await async_write_event(Synthesize(text=text, voice=None).event(), writer)
        
        chunk_count = 0
        
        while True:
            event = await async_read_event(reader)
            if event is None:
                break
            
            if AudioStart.is_type(event.type):
                audio_start = AudioStart.from_event(event)
                yield json.dumps({
                    "type": "audio_start",
                    "sample_rate": audio_start.rate,
                    "channels": audio_start.channels,
                    "sample_width": audio_start.width
                }) + "\n"
            
            elif AudioChunk.is_type(event.type):
                audio_chunk = AudioChunk.from_event(event)
                if audio_chunk.audio:
                    chunk_count += 1
                    audio_b64 = base64.b64encode(audio_chunk.audio).decode('utf-8')
                    yield json.dumps({
                        "type": "audio_chunk",
                        "data": audio_b64,
                        "chunk": chunk_count
                    }) + "\n"
                    _LOGGER.debug(f"Sent chunk {chunk_count}")
            
            elif AudioStop.is_type(event.type):
                yield json.dumps({"type": "audio_stop", "chunks": chunk_count}) + "\n"
                break
        
    except Exception as e:
        _LOGGER.error(f"Error: {e}", exc_info=True)
        yield json.dumps({"type": "error", "message": str(e)}) + "\n"
    
    finally:
        if writer:
            try:
                writer.close()
                await writer.wait_closed()
            except:
                pass


# HTML page with Web Audio API
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Qwen3-TTS Streaming</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 40px;
        }
        h1 {
            color: #667eea;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        .subtitle {
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            color: #333;
            font-weight: 600;
        }
        input, textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
            font-family: inherit;
            transition: border-color 0.3s;
        }
        input:focus, textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        textarea {
            resize: vertical;
            min-height: 120px;
        }
        .button-row {
            display: flex;
            gap: 10px;
            margin-bottom: 30px;
        }
        button {
            flex: 1;
            padding: 15px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        #startBtn {
            background: #667eea;
            color: white;
        }
        #startBtn:hover:not(:disabled) {
            background: #5568d3;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        #playBtn {
            background: #4CAF50;
            color: white;
        }
        #playBtn:hover:not(:disabled) {
            background: #45a049;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(76, 175, 80, 0.4);
        }
        #stopBtn {
            background: #f44336;
            color: white;
        }
        #stopBtn:hover:not(:disabled) {
            background: #da190b;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(244, 67, 54, 0.4);
        }
        .player {
            background: #f5f5f5;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }
        #status {
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            min-height: 60px;
            font-family: 'Courier New', monospace;
            border-left: 4px solid #667eea;
        }
        .stats {
            font-size: 14px;
            color: #666;
        }
        .emoji { font-size: 1.2em; }
    </style>
</head>
<body>
    <div class="container">
        <h1><span class="emoji">🎵</span> Qwen3-TTS Streaming</h1>
        <p class="subtitle">True seamless audio streaming with Web Audio API</p>
        
        <div class="form-group">
            <label for="host">Wyoming Server Host</label>
            <input type="text" id="host" value="localhost" />
        </div>
        
        <div class="form-group">
            <label for="port">Wyoming Server Port</label>
            <input type="number" id="port" value="10200" />
        </div>
        
        <div class="form-group">
            <label for="text">Text to Synthesize</label>
            <textarea id="text">Hello! This is true streaming audio. Listen as it plays seamlessly without any restarts or interruptions!</textarea>
        </div>
        
        <div class="button-row">
            <button id="startBtn">🎙️ Start Streaming</button>
        </div>
        
        <div class="player">
            <div id="status">Ready to stream...</div>
            <div class="button-row">
                <button id="playBtn" disabled>▶️ Play</button>
                <button id="stopBtn" disabled>⏹️ Stop</button>
            </div>
            <div class="stats">
                <span id="stats">Waiting for audio...</span>
            </div>
        </div>
    </div>

    <script>
        let audioContext = null;
        let nextPlayTime = 0;
        let isPlaying = false;
        let sampleRate = 16000;
        let channels = 1;
        let audioQueue = [];
        let receivedChunks = 0;
        let streamComplete = false;

        const startBtn = document.getElementById('startBtn');
        const playBtn = document.getElementById('playBtn');
        const stopBtn = document.getElementById('stopBtn');
        const status = document.getElementById('status');
        const stats = document.getElementById('stats');

        function updateStatus(msg) {
            status.textContent = msg;
        }

        function updateStats() {
            stats.textContent = `Chunks: ${receivedChunks}${streamComplete ? ' (complete)' : ''} | Playing: ${isPlaying ? 'Yes' : 'No'}`;
        }

        function base64ToArrayBuffer(base64) {
            const binaryString = atob(base64);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {
                bytes[i] = binaryString.charCodeAt(i);
            }
            return bytes.buffer;
        }

        function pcmToFloat32(pcmData) {
            const int16Array = new Int16Array(pcmData);
            const float32Array = new Float32Array(int16Array.length);
            for (let i = 0; i < int16Array.length; i++) {
                float32Array[i] = int16Array[i] / 32768.0;
            }
            return float32Array;
        }

        function scheduleAudioChunk(pcmData) {
            if (!audioContext) {
                audioContext = new (window.AudioContext || window.webkitAudioContext)({sampleRate: sampleRate});
                nextPlayTime = audioContext.currentTime;
            }
            
            const float32Data = pcmToFloat32(pcmData);
            const audioBuffer = audioContext.createBuffer(channels, float32Data.length, sampleRate);
            audioBuffer.getChannelData(0).set(float32Data);
            
            const source = audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(audioContext.destination);
            
            const playTime = Math.max(audioContext.currentTime, nextPlayTime);
            source.start(playTime);
            nextPlayTime = playTime + audioBuffer.duration;
        }

        function playAudio() {
            if (!isPlaying && audioQueue.length > 0) {
                isPlaying = true;
                playBtn.disabled = true;
                stopBtn.disabled = false;
                updateStatus('▶️ Playing audio...');
                
                while (audioQueue.length > 0) {
                    scheduleAudioChunk(audioQueue.shift());
                }
                updateStats();
            }
        }

        function stopAudio() {
            isPlaying = false;
            if (audioContext) {
                audioContext.close();
                audioContext = null;
            }
            audioQueue = [];
            nextPlayTime = 0;
            playBtn.disabled = false;
            stopBtn.disabled = true;
            updateStatus('⏹️ Stopped');
            updateStats();
        }

        async function startStreaming() {
            const text = document.getElementById('text').value;
            const host = document.getElementById('host').value;
            const port = document.getElementById('port').value;

            if (!text.trim()) {
                updateStatus('❌ Please enter some text');
                return;
            }

            receivedChunks = 0;
            streamComplete = false;
            audioQueue = [];
            startBtn.disabled = true;
            playBtn.disabled = true;

            try {
                updateStatus('🔌 Connecting...');
                
                const response = await fetch('/api/stream_audio', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({text, host, port})
                });

                const reader = response.body.getReader();
                const decoder = new TextDecoder();

                while (true) {
                    const {done, value} = await reader.read();
                    if (done) break;

                    const chunk = decoder.decode(value);
                    const lines = chunk.split('\\n');

                    for (const line of lines) {
                        if (!line.trim()) continue;

                        try {
                            const data = JSON.parse(line);

                            if (data.type === 'status') {
                                updateStatus(data.message);
                            } else if (data.type === 'audio_start') {
                                sampleRate = data.sample_rate;
                                channels = data.channels;
                                updateStatus(`🎵 Receiving audio: ${sampleRate}Hz`);
                            } else if (data.type === 'audio_chunk') {
                                receivedChunks++;
                                const pcmData = base64ToArrayBuffer(data.data);
                                
                                if (isPlaying) {
                                    scheduleAudioChunk(pcmData);
                                } else {
                                    audioQueue.push(pcmData);
                                    if (receivedChunks === 1) {
                                        playBtn.disabled = false;
                                        updateStatus('🎵 Ready! Click Play to start');
                                    }
                                }
                                updateStats();
                            } else if (data.type === 'audio_stop') {
                                streamComplete = true;
                                updateStatus(`✅ Stream complete! ${data.chunks} chunks`);
                                updateStats();
                                if (!isPlaying && audioQueue.length > 0) {
                                    playBtn.disabled = false;
                                }
                            } else if (data.type === 'error') {
                                updateStatus(`❌ Error: ${data.message}`);
                            }
                        } catch (e) {
                            console.error('Parse error:', e);
                        }
                    }
                }
            } catch (error) {
                updateStatus(`❌ Error: ${error.message}`);
            } finally {
                startBtn.disabled = false;
            }
        }

        startBtn.addEventListener('click', startStreaming);
        playBtn.addEventListener('click', playAudio);
        stopBtn.addEventListener('click', stopAudio);

        updateStatus('Ready to stream. Enter text and click Start!');
    </script>
</body>
</html>
"""


def create_app(wyoming_host: str = "localhost", wyoming_port: int = 10200):
    """Create FastAPI app."""
    app = FastAPI(title="Qwen3-TTS Streaming")
    
    @app.get("/", response_class=HTMLResponse)
    async def index():
        """Serve the HTML page."""
        return HTML_PAGE
    
    @app.post("/api/stream_audio")
    async def stream_audio(request: Request):
        """Stream audio endpoint."""
        data = await request.json()
        text = data.get("text", "")
        host = data.get("host", wyoming_host)
        port = int(data.get("port", wyoming_port))
        
        _LOGGER.info(f"Stream request: {text[:50]}... from {host}:{port}")
        
        return StreamingResponse(
            wyoming_audio_generator(text, host, port),
            media_type="text/event-stream"
        )
    
    return app


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description="Standalone Streaming UI")
    parser.add_argument("--wyoming-host", default="localhost")
    parser.add_argument("--wyoming-port", type=int, default=10200)
    parser.add_argument("--port", type=int, default=7860, help="Web server port")
    parser.add_argument("--host", default="0.0.0.0", help="Web server host")
    parser.add_argument("--debug", action="store_true")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    
    _LOGGER.info("Starting Standalone Streaming UI")
    _LOGGER.info(f"Web UI will be at http://{args.host}:{args.port}")
    _LOGGER.info(f"Wyoming server: {args.wyoming_host}:{args.wyoming_port}")
    
    app = create_app(args.wyoming_host, args.wyoming_port)
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
