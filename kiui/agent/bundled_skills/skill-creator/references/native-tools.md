# Kia Native Tools

Use `tools.py` when a skill benefits from a structured model-callable operation
or needs session-scoped in-process state. Prefer portable scripts for ordinary
one-shot commands.

## Contract

`tools.py` exposes a module-level `TOOLS` list. Each entry contains:

| Key | Required | Meaning |
|---|---:|---|
| `schema` | yes | OpenAI function schema. |
| `run` | yes | Callable invoked as `run(executor, **arguments)`. |
| `permission` | no | `safe` or `risky`; defaults to `risky`. |

`safe` avoids confirmation in default permission mode; strict mode prompts for
all tools. Mark mutating operations `risky`.

```python
def echo(executor, text: str) -> dict:
    if not text:
        return {"error": "text must not be empty", "success": False}
    return {"echoed": text, "success": True}


TOOLS = [
    {
        "permission": "safe",
        "run": echo,
        "schema": {
            "type": "function",
            "function": {
                "name": "echo_text",
                "description": "Echo non-empty text.",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        },
    }
]
```

Tool names must not collide with built-ins or other loaded skill tools. The
module is loaded standalone, so use absolute imports rather than sibling or
relative imports.

The schema guides the model, but kia does not guarantee runtime JSON Schema
validation. Validate constraints important to correctness. Return a dictionary
with a `success` boolean and keep results bounded.

Use `executor.console`, `executor._resolve_path(...)`, `executor.cancellation`,
and executor-owned state when needed. Do not assume optional packages such as
NumPy or Torch are installed; document dependencies in `compatibility`.

Loading `tools.py` executes it in the kia process. For this personal harness,
user-authored skills are normally trusted; review code obtained from elsewhere.

Mention enabled tools in `SKILL.md`, run the skill validator, then load the skill
and exercise each tool once.
