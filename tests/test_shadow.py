def test_utf8_roundtrip():
    from shadow.tokenizer import ShadowTokenizer

    tokenizer = ShadowTokenizer()
    text = "Hello — হ্যালো 你好"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_symbol_detection():
    from shadow.symbols import detect_symbols

    assert detect_symbols("5 + 7") == [{"symbol": "+", "type": "ADD"}]


def test_safe_calculation():
    from shadow.symbols import safe_calculate

    assert safe_calculate("10 + 20") == 30
    assert safe_calculate("(12 - 2) * 5") == 50


def test_restricted_calculation_rejects_calls():
    import pytest
    from shadow.symbols import safe_calculate

    with pytest.raises(SyntaxError):
        safe_calculate("__import__('os').system('echo unsafe')")


def test_bounded_memory():
    from shadow.memory import ShadowMemory

    memory = ShadowMemory(max_items=2)
    memory.add("a")
    memory.add("b")
    memory.add("c")
    assert memory.get() == ["b", "c"]


def test_model_forward_shape():
    import torch
    from shadow.model import ShadowModel

    model = ShadowModel(hidden_size=32, layers=1, heads=4)
    tokens = torch.tensor([[65, 66, 67]], dtype=torch.long)
    logits = model(tokens)
    assert logits.shape == (1, 3, 256)
