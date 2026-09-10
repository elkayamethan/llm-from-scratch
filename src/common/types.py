from typing import Literal

AttentionImpl = Literal["manual", "sdpa"]
History = dict[str, list[float | int]]
