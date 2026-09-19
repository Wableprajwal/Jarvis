import json


def parse_config(text: str) -> dict:
    """Parse a JSON config string; raise ValueError with a clear message on bad input."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid config: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ValueError("config must be a JSON object")
    data.setdefault("timeout_s", 30)
    return data
