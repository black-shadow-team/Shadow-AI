from collections import deque


class ShadowMemory:
    """A bounded working-memory buffer."""

    def __init__(self, max_items: int = 32):
        if max_items <= 0:
            raise ValueError("max_items must be positive")
        self.items: deque = deque(maxlen=max_items)

    def add(self, value) -> None:
        self.items.append(value)

    def get(self) -> list:
        return list(self.items)

    def clear(self) -> None:
        self.items.clear()
