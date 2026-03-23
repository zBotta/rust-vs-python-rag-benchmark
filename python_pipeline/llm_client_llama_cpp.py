"""LLM client — llama-cpp-python in-process backend (Task 15).

Runs inference in-process using llama-cpp-python with a GGUF model file.
No retry logic — in-process errors surface directly.
"""
from __future__ import annotations

import time

from python_pipeline.llm_client import LLMResponse, build_prompt

# Module-level cache: keyed by gguf_model_path so switching models still works.
_llama_cache: dict[str, object] = {}


def _get_llama(gguf_model_path: str):
    """Return a cached Llama instance for the given model path."""
    if gguf_model_path not in _llama_cache:
        from llama_cpp import Llama  # type: ignore[import]
        _llama_cache[gguf_model_path] = Llama(
            model_path=gguf_model_path, n_ctx=2048, verbose=False
        )
    return _llama_cache[gguf_model_path]


def generate(
    query: str,
    chunks: list[str],
    gguf_model_path: str,
    model: str | None = None,
    num_predict: int = 256,
) -> LLMResponse:
    """Run inference in-process via llama-cpp-python and return LLMResponse.

    Args:
        query: The user question.
        chunks: Retrieved context chunks.
        gguf_model_path: Path to the GGUF model file.
        model: Unused — present for API compatibility with llm_client.generate().
        num_predict: Maximum number of tokens to generate.

    Returns:
        LLMResponse with timing metrics and generated text.
    """
    try:
        llm = _get_llama(gguf_model_path)
        prompt = build_prompt(chunks, query)

        text_parts: list[str] = []
        ttft_ms: float = 0.0
        first_token = True
        last_chunk: dict | None = None

        start = time.perf_counter()

        for chunk in llm(prompt, max_tokens=num_predict, stream=True):
            token = chunk["choices"][0]["text"]
            if token and first_token:
                ttft_ms = (time.perf_counter() - start) * 1000.0
                first_token = False
            text_parts.append(token)
            last_chunk = chunk

        generation_ms = (time.perf_counter() - start) * 1000.0

        # Attempt to read token counts from the final chunk's usage field,
        # then fall back to llm.last_eval_tokens (completion tokens only).
        total_tokens: int = 0
        if last_chunk is not None:
            usage = last_chunk.get("usage")
            if usage:
                total_tokens = (usage.get("prompt_tokens", 0) or 0) + (
                    usage.get("completion_tokens", 0) or 0
                )

        if total_tokens == 0:
            # llama_cpp exposes last_eval_tokens as completion token count
            last_eval = getattr(llm, "last_eval_tokens", None)
            if last_eval is not None:
                total_tokens = int(last_eval)

        return LLMResponse(
            text="".join(text_parts),
            total_tokens=total_tokens,
            ttft_ms=ttft_ms,
            generation_ms=generation_ms,
            failed=False,
            failure_reason=None,
        )

    except Exception as exc:  # noqa: BLE001
        return LLMResponse(
            text="",
            total_tokens=0,
            ttft_ms=0.0,
            generation_ms=0.0,
            failed=True,
            failure_reason=str(exc),
        )
