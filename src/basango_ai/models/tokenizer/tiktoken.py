import tiktoken

from typing import Optional


class TikTokenTokenizer:
    def __init__(self, encoding: str = "cl100k_base", model: Optional[str] = None):
        self.encoding = encoding
        self.model = model
        self.tokenizer = (
            tiktoken.encoding_for_model(model)
            if model
            else tiktoken.get_encoding(encoding)
        )

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)

    def count(self, text: str) -> int:
        return len(self.encode(text))


def load_model() -> TikTokenTokenizer:
    return TikTokenTokenizer()
