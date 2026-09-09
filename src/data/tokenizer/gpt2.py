from collections.abc import Sequence

import tiktoken


class GPT2Tokenizer:
    """tiktoken wrapper. 'encode' never outputs special tokens"""

    def __init__(self) -> None:
        self._enc = tiktoken.get_encoding("gpt2")

    @property
    def vocab_size(self) -> int:
        return self._enc.n_vocab

    @property
    def eot_id(self) -> int:
        return self._enc.eot_token

    def encode(self, text: str) -> list[int]:
        return self._enc.encode_ordinary(text)

    def decode(self, ids: Sequence[int]) -> str:
        return self._enc.decode(list(ids))
