"""AeroMind AI: local-first, read-only retrieval augmented generation.

Python 3.12+. Core uses only the standard library. Optional:
  pip install pypdf sentence-transformers numpy

This module NEVER executes flight commands. Retrieved text is untrusted evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

LOG = logging.getLogger("aeromind.rag")
SUPPORTED = {".txt", ".md", ".markdown", ".json", ".jsonl", ".log", ".pdf"}
TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


@dataclass(frozen=True)
class RAGConfig:
    database_path: Path = Path("data/rag/knowledge.sqlite3")
    allowed_roots: tuple[Path, ...] = (Path("data/knowledge"),)
    chunk_size: int = 1200
    chunk_overlap: int = 200
    max_file_bytes: int = 15 * 1024 * 1024
    max_chunks_per_file: int = 1500
    max_context_chars: int = 9000
    top_k: int = 5
    embedding_model: str | None = None  # e.g. all-MiniLM-L6-v2
    ollama_model: str | None = None  # e.g. llama3.2
    ollama_url: str = "http://127.0.0.1:11434/api/generate"
    request_timeout_seconds: float = 90.0

    def __post_init__(self) -> None:
        if self.chunk_size < 100 or not (0 <= self.chunk_overlap < self.chunk_size):
            raise ValueError("Require chunk_size >= 100 and 0 <= overlap < chunk_size")
        if self.top_k < 1 or self.max_context_chars < 200:
            raise ValueError("top_k and context limits must be positive")


@dataclass(frozen=True)
class SearchHit:
    source: str
    chunk_index: int
    text: str
    score: float
    citation: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RAGAnswer:
    question: str
    answer: str
    sources: list[SearchHit]
    generated: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Chunk at paragraph/sentence/word boundaries when practical."""
    text = re.sub(r"\r\n?", "\n", text).strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            candidates = [text.rfind("\n\n", start + size // 2, end),
                          text.rfind(". ", start + size // 2, end),
                          text.rfind(" ", start + size // 2, end)]
            boundary = next((b for b in candidates if b > start), -1)
            if boundary > start:
                end = boundary + (2 if text[boundary:boundary+2] in ("\n\n", ". ") else 1)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)