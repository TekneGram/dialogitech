from __future__ import annotations

import json
import queue
import subprocess
import threading
from pathlib import Path
from typing import Any, Callable

class LLMWorker:
  def __init__(
      self,
      *,
      python_executable: str,
      runner_path: str | Path,
      model_path: str,
      request_timeout_seconds: float,
      event_logger: Callable[[str], None] | None = None,
  ) -> None:
    if request_timeout_seconds <= 0:
      raise ValueError("request_timeout_seconds must be positive.")

    self.request_timeout_seconds = request_timeout_seconds
    self.event_logger = event_logger or (lambda message: None)
    self._responses: queue.Queue[str | None] = queue.Queue()
    self._stderr_lines: list[str] = []
    self._stderr_lock = threading.Lock()

    self.process = subprocess.Popen(
      [python_executable, str(runner_path)],
      stdin=subprocess.PIPE,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      text=True,
      bufsize=1,
    )

    assert self.process.stdin is not None
    assert self.process.stdout is not None
    assert self.process.stderr is not None

    threading.Thread(
      target=self._read_stdout,
      args=(self.process.stdout,),
      daemon=True,
    ).start()

    threading.Thread(
      target=self._read_stderr,
      args=(self.process.stderr,),
      daemon=True,
    ).start()

    self._send(
      {
        "model_path": model_path,
        "messages": [],
        "max_tokens": 1,
        "temperature": 0.0,
        "initialize_only": True,
      }
    )

    self.event_logger("LLM worker started.")

  def generate(
      self,
      *,
      messages: list[dict[str, str]],
      max_tokens: int,
      temperature: float,
  ) -> str:
    self._send(
      {
        "model_path": "",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
      }
    )

    try:
      response_line = self._responses.get(
        timeout=self.request_timeout_seconds
      )
    except queue.Empty as exc:
      self.close()
      raise RuntimeError(
        "LLM worker request timed out. "
        f"Recent stderr: {self._recent_stderr()}"
      ) from exc

    if response_line is None:
      self.close()
      raise RuntimeError(
        "LLM worker exited unexpectedly. "
        f"Recent stderr: {self._recent_stderr()}"
      )

    try:
      payload = json.loads(response_line)
    except json.JSONDecodeError as exc:
      raise RuntimeError(
        f"LLM worker returned invalid JSON: {response_line!r}"
      ) from exc

    if payload.get("error"):
      raise RuntimeError(
        f"LLM error: {payload['error']}"
      )

    response = payload.get("response")
    if not isinstance(response, str) or not response.strip():
      raise RuntimeError("LLM worker returned an empty response")

    return response.strip()

  def close(self) -> None:
    process = getattr(self, "process", None)
    if process is None or process.poll() is not None:
      return

    self.event_logger("Stopping LLM worker.")

    try:
      if process.stdin is not None:
        process.stdin.close()
    except OSError:
      pass

    process.terminate()

    try:
      process.wait(timeout=5)
    except subprocess.TimeoutExpired:
      process.kill()
      process.wait()

  def __enter__(self) -> LLMWorker:
    return self

  def __exit__(
      self,
      exc_type: Any,
      exc_value: Any,
      traceback: Any
  ) -> None:
    self.close()

  def _send(self, payload: dict[str, object]) -> None:
    if self.process.poll() is not None:
      raise RuntimeError(
        f"LLM worker exited with code "
        f"{self.process.returncode}"
      )

    assert self.process.stdin is not None
    self.process.stdin.write(json.dumps(payload) + "\n")
    self.process.stdin.flush()

  def _read_stdout(self, stream: Any) -> None:
    for line in stream:
      self._responses.put(line.rstrip("\n"))

    self._responses.put(None)

  def _read_stderr(self, stream: Any) -> None:
    for line in stream:
      with self._stderr_lock:
        self._stderr_lines.append(line.rstrip("\n"))
        del self._stderr_lines[:-20]

  def _recent_stderr(self) -> str:
    with self._stderr_lock:
      return "\n".join(self._stderr_lines)
