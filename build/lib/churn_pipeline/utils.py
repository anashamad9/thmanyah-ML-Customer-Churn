from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    yaml = None


def load_structured_file(path: Path | str) -> Dict[str, Any]:
    """Load a tiny subset of YAML (or JSON) used for project configuration."""

    payload_path = Path(path)
    text = payload_path.read_text(encoding="utf-8")
    if yaml:
        return yaml.safe_load(text)
    return parse_simple_yaml(text)


def parse_simple_yaml(text: str) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    stack = [root]
    indents = [0]

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, _, raw_value = stripped.partition(":")
        key = key.strip()
        value = raw_value.strip()

        while indents and indent < indents[-1]:
            stack.pop()
            indents.pop()

        if value == "":
            new_dict: Dict[str, Any] = {}
            stack[-1][key] = new_dict
            stack.append(new_dict)
            indents.append(indent + 2)
            continue

        stack[-1][key] = _coerce_value(value)

    return root


def _coerce_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value
