"""Shared primitives for irlib retrieval algorithms.

Every retriever in the package follows the same reference contract:

Pseudocode:
    index(documents):
        normalize strings, dicts, and Document objects into Document records
        build the algorithm-specific search structure

    search(query, top_k):
        score candidate document ids
        return the top document ids and scores as list[(doc_id, score)]

The implementations are intentionally small and inspectable. They are useful
for experiments, tests, and teaching, while optional backends provide heavier
production-like behavior where appropriate.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence


TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


@dataclass()
class Document:
    """A normalized searchable document.

    Pseudocode:
        Document(id, text, metadata, fields)
        fields can hold title/body/tags for field-aware scoring

    Use this when callers need metadata, source paths, or structured fields
    while keeping search results as plain `(doc_id, score)` tuples.
    """

    id: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    fields: dict[str, str] = field(default_factory=dict)


class RetrieverProtocol(Protocol):
    """Minimal protocol shared by retrievers and wrappers."""

    documents: list[Document]

    def index(self, documents: Sequence[str | Mapping[str, Any] | Document]) -> None:
        """Build the retriever index."""

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Return ranked `(doc_id, score)` pairs."""


class BaseRetriever:
    """Base class that normalizes documents and exposes metadata lookup.

    Pseudocode:
        self.documents = normalize_documents(input_documents)
        subclasses build their own indexes over self.documents

    Subclasses should override `index` and `search`, but can call
    `_set_documents` to share normalization behavior.
    """

    def __init__(self) -> None:
        self.documents: list[Document] = []
        self._doc_by_id: dict[int, Document] = {}

    def _set_documents(self, documents: Sequence[str | Mapping[str, Any] | Document]) -> None:
        self.documents = normalize_documents(documents)
        self._doc_by_id = {doc.id: doc for doc in self.documents}

    def get_document(self, doc_id: int) -> Document:
        return self._doc_by_id[doc_id]

    def index(self, documents: Sequence[str | Mapping[str, Any] | Document]) -> None:
        self._set_documents(documents)

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        raise NotImplementedError


class DocumentStore:
    """Simple in-memory store used by loaders and retrievers.

    Pseudocode:
        add_many(documents)
        get(doc_id)
        iter_documents()

    The store keeps ids stable and does not impose a retrieval strategy.
    """

    def __init__(self, documents: Sequence[str | Mapping[str, Any] | Document] | None = None) -> None:
        self.documents: list[Document] = []
        self._doc_by_id: dict[int, Document] = {}
        if documents:
            self.add_many(documents)

    def add_many(self, documents: Sequence[str | Mapping[str, Any] | Document]) -> list[Document]:
        normalized = normalize_documents(documents, start_id=len(self.documents))
        for doc in normalized:
            self.documents.append(doc)
            self._doc_by_id[doc.id] = doc
        return normalized

    def get(self, doc_id: int) -> Document:
        return self._doc_by_id[doc_id]

    def __iter__(self) -> Iterable[Document]:
        return iter(self.documents)

    def __len__(self) -> int:
        return len(self.documents)


def normalize_documents(
    documents: Sequence[str | Mapping[str, Any] | Document],
    *,
    start_id: int = 0,
) -> list[Document]:
    """Normalize strings, dictionaries, and Document objects.

    Pseudocode:
        for item in documents:
            if item is Document: keep its fields
            if item is dict: read text, metadata, fields, id
            else: create Document from string
    """

    normalized: list[Document] = []
    for offset, item in enumerate(documents):
        default_id = start_id + offset
        if isinstance(item, Document):
            doc_id = item.id if item.id is not None else default_id
            normalized.append(
                Document(
                    id=int(doc_id),
                    text=item.text or "",
                    metadata=dict(item.metadata),
                    fields=dict(item.fields),
                )
            )
            continue

        if isinstance(item, Mapping):
            fields = {str(k): str(v) for k, v in dict(item.get("fields", {})).items()}
            metadata = dict(item.get("metadata", {}))
            doc_id = int(item.get("id", default_id))
            text = item.get("text")
            if text is None:
                text = "\n".join(fields.values())
            normalized.append(Document(id=doc_id, text=str(text or ""), metadata=metadata, fields=fields))
            continue

        normalized.append(Document(id=default_id, text=str(item), metadata={}, fields={}))
    return normalized


def tokenize(text: str, *, lowercase: bool = True) -> list[str]:
    """Tokenize text into word-like terms.

    Pseudocode:
        find all word spans
        lowercase each token when requested
    """

    if lowercase:
        text = text.lower()
    return TOKEN_RE.findall(text)


def token_positions(text: str) -> dict[str, list[int]]:
    """Return token positions for phrase and proximity retrieval."""

    positions: dict[str, list[int]] = {}
    for i, term in enumerate(tokenize(text)):
        positions.setdefault(term, []).append(i)
    return positions


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine similarity for dense or sparse vector values."""

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def sparse_cosine(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    """Cosine similarity for sparse dictionaries."""

    if len(a) > len(b):
        a, b = b, a
    dot = sum(value * b.get(term, 0.0) for term, value in a.items())
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def normalize_scores(scores: Mapping[int, float]) -> dict[int, float]:
    """Min-max normalize a score dictionary to `[0, 1]`."""

    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if math.isclose(lo, hi):
        return {doc_id: 1.0 for doc_id in scores}
    return {doc_id: (score - lo) / (hi - lo) for doc_id, score in scores.items()}


def top_k(scores: Mapping[int, float], k: int) -> list[tuple[int, float]]:
    """Return sorted top-k `(doc_id, score)` pairs."""

    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]


def word_ngrams(text: str, n: int) -> list[tuple[str, ...]]:
    tokens = tokenize(text)
    if n <= 0:
        return []
    return [tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1))]


def char_ngrams(text: str, n: int = 3) -> list[str]:
    compact = re.sub(r"\s+", " ", text.lower()).strip()
    if len(compact) < n:
        return [compact] if compact else []
    return [compact[i : i + n] for i in range(len(compact) - n + 1)]


def chunk_document(document: Document, *, chunk_size: int = 300, overlap: int = 50) -> list[Document]:
    """Split a document into overlapping word chunks.

    Pseudocode:
        words = tokenize preserving original word strings
        step = chunk_size - overlap
        for each window create a child Document with parent metadata
    """

    words = re.findall(r"\S+", document.text)
    if not words or len(words) <= chunk_size:
        return [document]
    step = max(1, chunk_size - overlap)
    chunks: list[Document] = []
    for chunk_index, start in enumerate(range(0, len(words), step)):
        window = words[start : start + chunk_size]
        if not window:
            continue
        metadata = dict(document.metadata)
        metadata.update({"parent_id": document.id, "chunk_id": chunk_index, "chunk_start_word": start})
        chunks.append(
            Document(
                id=document.id * 100000 + chunk_index,
                text=" ".join(window),
                metadata=metadata,
                fields=dict(document.fields),
            )
        )
        if start + chunk_size >= len(words):
            break
    return chunks


def chunk_documents(
    documents: Sequence[Document],
    *,
    chunk_size: int = 300,
    overlap: int = 50,
) -> list[Document]:
    chunks: list[Document] = []
    for doc in documents:
        chunks.extend(chunk_document(doc, chunk_size=chunk_size, overlap=overlap))
    return chunks


def read_text_lossy(path: str | Path) -> str:
    """Read a text-like file with safe fallback encodings."""

    data = Path(path).read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def optional_import(module_name: str, package_hint: str | None = None) -> Any:
    """Import an optional dependency with a clear error message."""

    try:
        return __import__(module_name)
    except ImportError as exc:
        hint = package_hint or module_name
        raise ImportError(f"Install `{hint}` to use this feature.") from exc

