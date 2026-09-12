from __future__ import annotations

import inspect
import json
import sys


def main() -> None:
    from mlx_lm import generate, load

    first_line = sys.stdin.readline()
    if not first_line:
        return
    first_payload = json.loads(first_line)
    model_path = first_payload["model_path"]
    model, tokenizer = load(model_path)

    # The first line is used to discover the model path and load the model.
    # Do not materialize sys.stdin with list expansion: the parent keeps the
    # pipe open for subsequent requests, so waiting for EOF would deadlock.
    if not first_payload.get("initialize_only"):
        _handle_request(
            payload=first_payload,
            model=model,
            tokenizer=tokenizer,
            generate=generate,
        )

    for line in sys.stdin:
        try:
            payload = json.loads(line)
            if payload.get("initialize_only"):
                continue
            _handle_request(
                payload=payload,
                model=model,
                tokenizer=tokenizer,
                generate=generate,
            )
        except Exception as exc:  # pragma: no cover - exercised in external MLX runtime
            print(f"Gemma worker request failed: {exc}", file=sys.stderr, flush=True)
            sys.stdout.write(json.dumps({"error": str(exc)}) + "\n")
            sys.stdout.flush()


def _handle_request(*, payload: dict[str, object], model: object, tokenizer: object, generate: object) -> None:
    messages = payload["messages"]
    max_tokens = int(payload.get("max_tokens", 220))
    temperature = float(payload.get("temperature", 0.0))

    try:
        prompt = tokenizer.apply_chat_template(  # type: ignore[attr-defined]
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except AttributeError:
        rendered: list[str] = []
        for message in messages:  # type: ignore[union-attr]
            rendered.append(f"{message['role'].upper()}:\n{message['content']}")
        rendered.append("ASSISTANT:")
        prompt = "\n\n".join(rendered)

    generate_kwargs = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "verbose": False,
    }
    parameter_names = set(inspect.signature(generate).parameters)  # type: ignore[arg-type]
    if "temp" in parameter_names:
        generate_kwargs["temp"] = temperature
    elif "temperature" in parameter_names:
        generate_kwargs["temperature"] = temperature

    response = generate(model, tokenizer, **generate_kwargs)  # type: ignore[operator]
    sys.stdout.write(json.dumps({"response": response.strip()}) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
