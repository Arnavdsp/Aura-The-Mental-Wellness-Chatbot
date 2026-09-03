"""Gemma 3n backend.

Gemma 3n is natively multimodal — text, images and audio all go in through the
same chat template — so this engine passes media straight to the processor
rather than bolting separate captioning and ASR models onto the side.

Weights load lazily in a worker thread so the event loop is never blocked, and
generation runs in a thread too (``transformers`` is synchronous). Unsloth is
used when present for 4-bit loading; otherwise plain ``transformers`` is used.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from queue import Empty, Queue
from typing import Any

from aura.config import Settings
from aura.engine.base import CoachEngine, GenerationRequest
from aura.logging import get_logger

log = get_logger(__name__)

_SENTINEL = object()


class GemmaEngine(CoachEngine):
    name = "gemma"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Any = None
        self._processor: Any = None
        self._device = settings.device
        self._lock = asyncio.Lock()
        # transformers generation is not re-entrant across threads for one model.
        self._generate_lock = threading.Lock()
        self._ready = False

    # -- lifecycle -------------------------------------------------------

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def capabilities(self) -> dict[str, bool]:
        return {
            "text": True,
            "vision": self._settings.enable_vision,
            "audio_in": self._settings.enable_audio_input
            and self._settings.asr_backend == "gemma",
            "streaming": True,
        }

    async def warmup(self) -> None:
        if self._ready:
            return
        async with self._lock:
            if self._ready:
                return
            await asyncio.to_thread(self._load)
            self._ready = True

    def _load(self) -> None:
        import torch  # type: ignore[import-not-found]

        settings = self._settings
        self._device = settings.device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info("loading %s on %s", settings.model_id, self._device)

        model = processor = None
        try:  # Unsloth gives us fast 4-bit loading when it is installed.
            from unsloth import FastModel  # type: ignore[import-not-found]

            model, processor = FastModel.from_pretrained(
                model_name=settings.model_id,
                dtype=None,
                max_seq_length=settings.max_seq_length,
                load_in_4bit=settings.load_in_4bit,
                full_finetuning=False,
            )
            log.info("loaded via unsloth")
        except Exception as exc:  # pragma: no cover - env dependent
            log.info("unsloth unavailable (%s); using transformers directly", exc)

        if model is None:
            from transformers import AutoModelForImageTextToText, AutoProcessor

            processor = AutoProcessor.from_pretrained(settings.model_id)
            model = AutoModelForImageTextToText.from_pretrained(
                settings.model_id,
                torch_dtype=torch.bfloat16 if self._device == "cuda" else torch.float32,
                device_map="auto" if self._device == "cuda" else None,
            )
            if self._device != "cuda":
                model = model.to(self._device)

        if settings.adapter_path:
            from peft import PeftModel  # type: ignore[import-not-found]

            model = PeftModel.from_pretrained(model, settings.adapter_path)
            log.info("applied LoRA adapter from %s", settings.adapter_path)

        model.eval()
        self._model, self._processor = model, processor
        log.info("model ready")

    async def shutdown(self) -> None:
        self._model = self._processor = None
        self._ready = False
        try:
            import torch  # type: ignore[import-not-found]

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # -- prompt assembly -------------------------------------------------

    def _build_messages(self, request: GenerationRequest) -> list[dict[str, Any]]:
        """Gemma 3n chat format: every content part is a typed dict."""
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": [{"type": "text", "text": request.system_prompt}],
            }
        ]
        for entry in request.history:
            messages.append(
                {
                    "role": entry["role"],
                    "content": [{"type": "text", "text": entry["content"]}],
                }
            )

        parts: list[dict[str, Any]] = []
        for image in request.images:
            parts.append({"type": "image", "image": _to_pil(image)})
        for audio in request.audio:
            parts.append({"type": "audio", "audio": audio})
        if request.user_text:
            parts.append({"type": "text", "text": request.user_text})
        if not parts:
            parts.append({"type": "text", "text": "(no message)"})

        messages.append({"role": "user", "content": parts})
        return messages

    def _prepare_inputs(self, request: GenerationRequest) -> Any:
        messages = self._build_messages(request)
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        return inputs.to(self._model.device)

    def _generation_kwargs(self, request: GenerationRequest) -> dict[str, Any]:
        settings = self._settings
        temperature = (
            request.temperature if request.temperature is not None else settings.temperature
        )
        return {
            "max_new_tokens": request.max_new_tokens or settings.max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": max(temperature, 1e-5),
            "top_p": settings.top_p,
            "top_k": settings.top_k,
            "repetition_penalty": settings.repetition_penalty,
        }

    # -- inference -------------------------------------------------------

    async def generate(self, request: GenerationRequest) -> str:
        await self.warmup()
        return await asyncio.to_thread(self._generate_sync, request)

    def _generate_sync(self, request: GenerationRequest) -> str:
        import torch  # type: ignore[import-not-found]

        with self._generate_lock:
            inputs = self._prepare_inputs(request)
            prompt_length = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                output = self._model.generate(**inputs, **self._generation_kwargs(request))
            new_tokens = output[0][prompt_length:]
            text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text.strip()

    async def stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        """Bridge the synchronous ``TextIteratorStreamer`` onto the event loop."""
        await self.warmup()
        from transformers import TextIteratorStreamer

        queue: Queue[Any] = Queue()
        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        def run() -> None:
            import torch  # type: ignore[import-not-found]

            try:
                with self._generate_lock:
                    inputs = self._prepare_inputs(request)
                    with torch.inference_mode():
                        self._model.generate(
                            **inputs, streamer=streamer, **self._generation_kwargs(request)
                        )
            except Exception as exc:  # surfaced to the consumer below
                queue.put(exc)
            finally:
                queue.put(_SENTINEL)

        worker = threading.Thread(target=run, name="gemma-generate", daemon=True)
        worker.start()

        loop = asyncio.get_running_loop()
        iterator = iter(streamer)
        try:
            while True:
                chunk = await loop.run_in_executor(None, _next_chunk, iterator, queue)
                if chunk is _SENTINEL:
                    break
                if isinstance(chunk, Exception):
                    raise chunk
                if chunk:
                    yield chunk
        finally:
            await loop.run_in_executor(None, worker.join, 5.0)

    async def transcribe(self, audio: bytes, media_type: str) -> str:
        """Use Gemma 3n's own audio encoder rather than a separate ASR model."""
        await self.warmup()
        request = GenerationRequest(
            system_prompt="You are a precise transcription engine.",
            user_text=(
                "Transcribe the spoken audio verbatim. Output only the transcript, "
                "with no commentary, labels or quotation marks."
            ),
            audio=[audio],
            temperature=0.0,
            max_new_tokens=512,
        )
        return await asyncio.to_thread(self._generate_sync, request)

    @property
    def _tokenizer(self) -> Any:
        # Processors expose the tokenizer differently across versions.
        return getattr(self._processor, "tokenizer", self._processor)


def _next_chunk(iterator: Any, queue: Queue[Any]) -> Any:
    """Pull the next chunk, preferring a generation error over a hang."""
    try:
        return next(iterator)
    except StopIteration:
        try:
            return queue.get(timeout=1.0)
        except Empty:
            return _SENTINEL


def _to_pil(data: bytes) -> Any:
    import io

    from PIL import Image  # type: ignore[import-not-found]

    return Image.open(io.BytesIO(data)).convert("RGB")
