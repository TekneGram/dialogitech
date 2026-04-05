from __future__ import annotations

import inspect
import json
import sys


def main() -> None:
    payload = json.load(sys.stdin)
    model_path = payload["model_path"]
    messages = payload["messages"]
    max_tokens = int(payload.get("max_tokens", 220))
    temperature = float(payload.get("temperature", 0.0))

    from mlx_lm import generate, load

    model, tokenizer = load(model_path)

    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except AttributeError:
        rendered: list[str] = []
        for message in messages:
            rendered.append(f"{message['role'].upper()}:\n{message['content']}")
        rendered.append("ASSISTANT:")
        prompt = "\n\n".join(rendered)

    generate_kwargs = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "verbose": False,
    }
    parameter_names = set(inspect.signature(generate).parameters)
    if "temp" in parameter_names:
        generate_kwargs["temp"] = temperature
    elif "temperature" in parameter_names:
        generate_kwargs["temperature"] = temperature

    response = generate(model, tokenizer, **generate_kwargs)
    sys.stdout.write(response.strip())


if __name__ == "__main__":
    main()
