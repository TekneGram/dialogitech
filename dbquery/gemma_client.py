from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Callable


class GemmaClient:
    def __init__(
        self,
        *,
        model_path: str,
        python_executable: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        event_logger: Callable[[str], None] | None = None,
    ) -> None:
        self.model_path = model_path
        self.python_executable = python_executable
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.event_logger = event_logger

    def generate(self, messages: list[dict[str, str]], *, max_tokens: int | None = None) -> str:
        runner_path = Path(__file__).resolve().parent.parent / "chunker" / "mlx_llm_runner.py"
        payload = {
            "model_path": self.model_path,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": self.temperature,
        }
        completed = subprocess.run(
            [self.python_executable, str(runner_path)],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=False,
        )

        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            raise RuntimeError(
                f"External MLX runner failed with exit code {completed.returncode}: {stderr or 'no stderr output'}"
            )

        raw_response = completed.stdout.strip()
        if not raw_response:
            raise RuntimeError("External MLX runner returned empty output.")

        response = self._extract_runner_response(raw_response)
        self._log_event(f"Gemma raw response: {self._truncate_for_log(response)}")
        return response

    def _extract_runner_response(self, raw_response: str) -> str:
        """Unwrap the JSON line protocol used by ``mlx_llm_runner``.

        Older external runners may return plain model text, so preserve that
        behavior when stdout is not a runner envelope.
        """
        try:
            payload = json.loads(raw_response)
        except json.JSONDecodeError:
            return raw_response

        if not isinstance(payload, dict):
            return raw_response
        if "error" in payload:
            raise RuntimeError(f"External MLX runner failed: {payload['error']}")
        if "response" not in payload:
            return raw_response

        response = payload["response"]
        if not isinstance(response, str) or not response.strip():
            raise RuntimeError("External MLX runner returned an empty response.")
        return response.strip()

    def _log_event(self, message: str) -> None:
        if self.event_logger is not None:
            self.event_logger(message)

    def _truncate_for_log(self, response: str, max_chars: int = 240) -> str:
        collapsed = " ".join(response.split())
        if len(collapsed) <= max_chars:
            return collapsed
        return f"{collapsed[: max_chars - 3]}..."
