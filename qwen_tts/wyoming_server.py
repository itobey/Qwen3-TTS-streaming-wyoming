#!/usr/bin/env python3
"""Wyoming protocol TTS server for Qwen3-TTS."""

import argparse
import asyncio
import io
import logging
import os
import wave
from functools import partial
from pathlib import Path

import numpy as np
import torch

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import Synthesize, SynthesizeChunk, SynthesizeStart, SynthesizeStop, SynthesizeStopped

from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem
from dataclasses import asdict

_LOGGER = logging.getLogger(__name__)

# Language code mapping from ISO 639-1 codes to full language names
LANGUAGE_CODE_MAP = {
    "en": "english",
    "zh": "chinese",
    "fr": "french",
    "de": "german",
    "it": "italian",
    "ja": "japanese",
    "ko": "korean",
    "pt": "portuguese",
    "ru": "russian",
    "es": "spanish",
}


class Qwen3TTSEventHandler(AsyncEventHandler):
    """Event handler for Wyoming protocol."""

    def __init__(
        self,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        model: Qwen3TTSModel,
        voice_prompts: dict,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.model = model
        self.voice_prompts = voice_prompts
        self.is_streaming: bool = False
        self._synthesize_voice = None
        self._text_buffer = ""

    async def handle_event(self, event: Event) -> bool:
        """Handle Wyoming protocol events."""
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        if SynthesizeStart.is_type(event.type):
            # Start of text stream from conversation agent
            stream_start = SynthesizeStart.from_event(event)
            self.is_streaming = True
            self._synthesize_voice = stream_start.voice
            self._text_buffer = ""
            _LOGGER.info("Text stream started")
            return True

        if SynthesizeChunk.is_type(event.type):
            # Received text chunk from conversation agent
            stream_chunk = SynthesizeChunk.from_event(event)
            self._text_buffer += stream_chunk.text
            
            # Check for sentence boundaries and synthesize complete sentences
            sentences = self._extract_sentences(self._text_buffer)
            for sentence in sentences:
                _LOGGER.info(f"Synthesizing sentence: {sentence[:50]}...")
                await self._synthesize_text(sentence, self._synthesize_voice, send_stop=False)
            
            return True

        if SynthesizeStop.is_type(event.type):
            # End of text stream - synthesize remaining text
            if self._text_buffer.strip():
                _LOGGER.info(f"Synthesizing final text: {self._text_buffer[:50]}...")
                await self._synthesize_text(self._text_buffer, self._synthesize_voice, send_stop=True)
            else:
                await self.write_event(AudioStop().event())
            
            await self.write_event(SynthesizeStopped().event())
            self.is_streaming = False
            self._text_buffer = ""
            _LOGGER.info("Text stream stopped")
            return True

        if Synthesize.is_type(event.type):
            if self.is_streaming:
                # Ignore - we're in streaming mode
                return True
            
            # Non-streaming mode - synthesize entire text at once
            synthesize = Synthesize.from_event(event)
            _LOGGER.info(f"Synthesizing (non-streaming): {synthesize.text[:50]}...")
            await self._synthesize_text(synthesize.text, synthesize.voice, send_stop=True)
            return True

        return True
    
    def _extract_sentences(self, text: str) -> list:
        """Extract complete sentences from text buffer and update buffer."""
        sentence_endings = '.!?。！？'
        sentences = []
        
        while text:
            # Find the last sentence ending
            last_ending = -1
            for i, char in enumerate(text):
                if char in sentence_endings:
                    last_ending = i
            
            if last_ending == -1:
                # No complete sentence yet
                break
            
            # Extract sentence (including the punctuation)
            sentence = text[:last_ending + 1].strip()
            if sentence:
                sentences.append(sentence)
            
            # Update buffer
            text = text[last_ending + 1:].strip()
        
        self._text_buffer = text
        return sentences
    
    async def _synthesize_text(self, text: str, voice, send_stop: bool = True) -> None:
        """Synthesize text and send audio chunks."""
        if not text.strip():
            if send_stop:
                await self.write_event(AudioStop().event())
            return
        
        # Get voice settings
        voice_id = voice.name if voice else (self.cli_args.default_voice or "default")
        voice_prompt = self.voice_prompts.get(voice_id)
        
        # Determine language
        language = self.cli_args.language
        if voice:
            # Check for language (singular) or languages (plural, list)
            if hasattr(voice, 'language') and voice.language:
                language = voice.language
            elif hasattr(voice, 'languages') and voice.languages:
                # Take the first language from the list
                language = voice.languages[0]
        
        # Map language code to full name if needed
        if language and language.lower() in LANGUAGE_CODE_MAP:
            mapped_language = LANGUAGE_CODE_MAP[language.lower()]
            _LOGGER.debug(f"Mapped language code '{language}' to '{mapped_language}'")
            language = mapped_language
        
        try:
            # Generate audio using streaming
            sample_rate = None
            audio_start_sent = False
            chunk_count = 0
            
            _LOGGER.info("Starting stream generation...")
            for chunk, sr in self.model.stream_generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=voice_prompt,
                emit_every_frames=self.cli_args.emit_every_frames,
                decode_window_frames=self.cli_args.decode_window_frames,
                first_chunk_emit_every=self.cli_args.first_chunk_emit_every,
                first_chunk_decode_window=self.cli_args.first_chunk_decode_window,
                first_chunk_frames=self.cli_args.first_chunk_frames,
            ):
                chunk_count += 1
                _LOGGER.info(f"Received chunk {chunk_count} from model ({len(chunk)} samples)")
                
                # Send AudioStart on first chunk
                if not audio_start_sent:
                    sample_rate = sr
                    await self.write_event(
                        AudioStart(
                            rate=sample_rate,
                            width=2,
                            channels=1,
                        ).event()
                    )
                    audio_start_sent = True
                    _LOGGER.info(f"Sent AudioStart at {sample_rate} Hz")
                
                # Convert chunk to 16-bit PCM
                if chunk.dtype == np.float32 or chunk.dtype == np.float64:
                    chunk = np.clip(chunk, -1.0, 1.0)
                    chunk = (chunk * 32767).astype(np.int16)
                
                # Send chunk immediately
                chunk_bytes = chunk.tobytes()
                await self.write_event(
                    AudioChunk(
                        audio=chunk_bytes,
                        rate=sample_rate,
                        width=2,
                        channels=1,
                    ).event()
                )
                _LOGGER.info(f"Sent AudioChunk {chunk_count}: {len(chunk_bytes)} bytes")
            
            if send_stop:
                await self.write_event(AudioStop().event())
            _LOGGER.info(f"Synthesis complete - sent {chunk_count} chunks total")
            
        except Exception as e:
            _LOGGER.exception(f"Error synthesizing: {e}")
            raise


async def async_main() -> None:
    """Start Wyoming TTS server."""
    parser = argparse.ArgumentParser(description="Qwen3-TTS Wyoming Protocol Server")
    parser.add_argument(
        "--uri",
        default="tcp://0.0.0.0:10200",
        help="URI to bind server (default: tcp://0.0.0.0:10200)",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        help="Model name from HuggingFace Hub (default: Qwen/Qwen3-TTS-12Hz-1.7B-Base)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available)",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Default language code (default: en)",
    )
    parser.add_argument(
        "--default-voice",
        help="Default voice name",
    )
    parser.add_argument(
        "--default-voice-ref",
        help="Path to reference audio file for default voice",
    )
    parser.add_argument(
        "--default-voice-text",
        help="Transcript of reference audio for default voice",
    )
    parser.add_argument(
        "--default-voice-pt",
        help="Path to .pt file containing pre-extracted voice embeddings (alternative to --default-voice-ref)",
    )
    parser.add_argument(
        "--emit-every-frames",
        type=int,
        default=12,
        help="Emit audio every N frames (default: 12)",
    )
    parser.add_argument(
        "--decode-window-frames",
        type=int,
        default=80,
        help="Decoder context window (default: 80)",
    )
    parser.add_argument(
        "--first-chunk-emit-every",
        type=int,
        default=5,
        help="Phase 1 emit interval (default: 5, 0 to disable two-phase)",
    )
    parser.add_argument(
        "--first-chunk-decode-window",
        type=int,
        default=48,
        help="Phase 1 decode window (default: 48)",
    )
    parser.add_argument(
        "--first-chunk-frames",
        type=int,
        default=48,
        help="Switch to phase 2 after N frames (default: 48)",
    )
    parser.add_argument(
        "--enable-optimizations",
        action="store_true",
        help="Enable torch.compile and CUDA graph optimizations",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    
    _LOGGER.info(f"Loading model from HuggingFace Hub...")
    _LOGGER.info(f"Model: {args.model}")
    _LOGGER.info(f"This will download the model if not already cached (requires ~4GB)")
    
    # Use the simple from_pretrained - it will download if needed
    model = Qwen3TTSModel.from_pretrained(
        args.model,
        dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
        device_map=args.device,
    )
    
    if args.enable_optimizations:
        _LOGGER.info("Enabling streaming optimizations...")
        # Configure CUDA graphs for dynamic shapes
        # This avoids warnings about recording too many graphs and can improve performance
        torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit = None
        
        model.enable_streaming_optimizations(
            decode_window_frames=args.decode_window_frames,
            use_compile=True,
            compile_mode="reduce-overhead",
        )
    
    # Load default voice if specified
    voice_prompts = {}
    if args.default_voice_pt:
        # Load from .pt file (pre-extracted embeddings)
        voice_name = args.default_voice or "default"
        _LOGGER.info(f"Loading default voice from .pt file: {voice_name}")
        payload = torch.load(args.default_voice_pt, map_location="cpu", weights_only=True)
        items_data = payload["items"]
        
        # Reconstruct VoiceClonePromptItem objects
        voice_prompts_list = []
        for d in items_data:
            ref_code = torch.tensor(d["ref_code"]) if not torch.is_tensor(d["ref_code"]) else d["ref_code"]
            ref_spk = torch.tensor(d["ref_spk_embedding"]) if not torch.is_tensor(d["ref_spk_embedding"]) else d["ref_spk_embedding"]
            
            voice_prompts_list.append(
                VoiceClonePromptItem(
                    ref_code=ref_code,
                    ref_spk_embedding=ref_spk,
                    x_vector_only_mode=d.get("x_vector_only_mode", False),
                    icl_mode=d.get("icl_mode", True),
                    ref_text=d.get("ref_text")
                )
            )
        voice_prompts[voice_name] = voice_prompts_list
    elif args.default_voice_ref:
        # Load from audio file (extract embeddings at runtime)
        voice_name = args.default_voice or "default"
        _LOGGER.info(f"Loading default voice from audio file: {voice_name}")
        voice_prompts[voice_name] = model.create_voice_clone_prompt(
            ref_audio=args.default_voice_ref,
            ref_text=args.default_voice_text or "",
        )
    
    # Build Wyoming Info
    wyoming_info = Info(
        tts=[
            TtsProgram(
                name="qwen3-tts",
                description="Qwen3-TTS streaming text-to-speech",
                attribution=Attribution(
                    name="Alibaba Qwen Team",
                    url="https://github.com/QwenLM/Qwen3-TTS",
                ),
                installed=True,
                version="1.0",
                supports_synthesize_streaming=True,
                voices=[
                    TtsVoice(
                        name=args.default_voice or "default",
                        description=f"Qwen3-TTS voice ({args.language})",
                        attribution=Attribution(
                            name="Alibaba Qwen Team",
                            url="https://github.com/QwenLM/Qwen3-TTS",
                        ),
                        installed=True,
                        languages=[args.language],
                        version="1.0",
                    )
                ],
            )
        ],
    )
    
    _LOGGER.info(f"Starting Wyoming server on {args.uri}")
    
    server = AsyncServer.from_uri(args.uri)
    
    await server.run(
        partial(
            Qwen3TTSEventHandler,
            wyoming_info,
            args,
            model,
            voice_prompts,
        )
    )


def main() -> None:
    """Entry point for console script."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
