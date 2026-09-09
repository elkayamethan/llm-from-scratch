import yaml

from pathlib import Path
from typing import TypeVar
from pydantic import BaseModel


_M = TypeVar("_M", bound=BaseModel)

def load_config(path: str | Path, cls: type[_M]) -> _M:
    """Returns an instance of cls validated from the YAML mapping at path."""

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping, received: {type(data).__name__}")

    return cls.model_validate(data)
