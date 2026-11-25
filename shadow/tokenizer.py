class ShadowTokenizer:
    """Minimal UTF-8 byte tokenizer."""

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        if not tokens:
            return ""
        return bytes(tokens).decode("utf-8", errors="replace")
