import ast
import operator

SYMBOLS = {
    "+": "ADD",
    "-": "SUBTRACT",
    "*": "MULTIPLY",
    "/": "DIVIDE",
    "=": "EQUAL",
    ">": "GREATER",
    "<": "LESS",
}

OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def detect_symbols(text: str) -> list[dict[str, str]]:
    return [
        {"symbol": character, "type": SYMBOLS[character]}
        for character in text
        if character in SYMBOLS
    ]


def safe_calculate(expression: str) -> float | int:
    """Evaluate a deliberately restricted arithmetic expression tree."""
    tree = ast.parse(expression, mode="eval")

    def evaluate(node: ast.AST) -> float | int:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("Only numeric constants are supported")
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        raise ValueError("Unsupported expression")

    return evaluate(tree.body)
