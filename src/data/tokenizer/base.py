from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    """Structural interface for a text tokenizer"""

    @property
    def vocab_size(self) -> int:
        """Number of unique token ids"""
        ...

    def encode(self, text: str) -> list[int]:
        """Converts 'text' into a sequence of token ids"""
        ...

    def decode(self, ids: Sequence[int]) -> str:
        """Converts a sequence of token ids back into text"""
        ...
