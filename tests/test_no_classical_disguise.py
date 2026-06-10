from __future__ import annotations
import ast
from pathlib import Path

FORBIDDEN = {"Linear", "Embedding", "GRU", "LSTM", "Transformer", "MultiheadAttention"}
ROOT = Path(__file__).resolve().parents[1]

def _attr_name(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))

def test_no_forbidden_module_calls():
    hits = []
    for py in ROOT.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        tree = ast.parse(py.read_text(encoding="utf-8", errors="ignore"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _attr_name(node.func)
                if name.split(".")[-1] in FORBIDDEN:
                    hits.append((str(py.relative_to(ROOT)), getattr(node, "lineno", None), name))
    assert hits == []
