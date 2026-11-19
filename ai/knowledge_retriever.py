
"""AeroMind AI — Knowledge Retriever.

Local Retrieval-Augmented Generation (RAG) retrieval layer.

Features:
- PDF, TXT, Markdown and JSON ingestion
- Semantic search using Sentence Transformers
- Persistent ChromaDB vector storage
- Document chunking and overlap
- Incremental document indexing
- Category-based retrieval
- Source references and logging

This module retrieves information only. It does not issue flight commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

LOGGER = logging.getLogger("aeromind.ai.knowledge_retriever")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge_base"
DEFAULT_DB_DIR = PROJECT_ROOT / "vector_database"

CATEGORIES = frozenset({
    "flight_manuals",
    "safety_rules",
    "mission_history",
    "object_information",
})

SUPPORTED_EXTENSIONS = frozenset({
    ".txt", ".md", ".pdf", ".json"
})


class KnowledgeRetrieverError(RuntimeError):
    """Error raised during knowledge retrieval or indexing."""


@dataclass(frozen=True)
class RetrievalResult:
    """A retrieved document passage with source information."""

    text: str
    source: str
    category: str
    chunk_index: int
    distance: float
    document_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source,
            "category": self.category,
            "chunk_index": self.chunk_index,
            "distance": self.distance,
            "document_id": self.document_id,
            "metadata": dict(self.metadata),
        }


class KnowledgeRetriever:
    """Persistent semantic document retrieval for AeroMind AI."""

    def __init__(
        self,
        knowledge_dir: str | Path = DEFAULT_KNOWLEDGE_DIR,
        database_dir: str | Path = DEFAULT_DB_DIR,
        collection_name: str = "aeromind_knowledge_v1",
        embedding_model: str = (
            "sentence-transformers/all-MiniLM-L6-v2"
        ),
        chunk_size: int = 900,
        chunk_overlap: int = 150,
        batch_size: int = 64,
    ) -> None:

        if chunk_size < 100:
            raise ValueError("chunk_size must be at least 100")

        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("Invalid chunk overlap")

        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        if not re.fullmatch(
            r"[a-zA-Z][a-zA-Z0-9_-]{2,62}",
            collection_name,
        ):
            raise ValueError("Invalid collection name")

        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise KnowledgeRetrieverError(
                "Install dependencies using: "
                "pip install chromadb sentence-transformers pypdf"
            ) from exc

        self.knowledge_dir = (
            Path(knowledge_dir).expanduser().resolve()
        )
        self.database_dir = (
            Path(database_dir).expanduser().resolve()
        )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size

        self.knowledge_dir.mkdir(
            parents=True, exist_ok=True
        )
        self.database_dir.mkdir(
            parents=True, exist_ok=True
        )

        for category in CATEGORIES:
            (self.knowledge_dir / category).mkdir(
                exist_ok=True
            )

        LOGGER.info(
            "Loading embedding model: %s",
            embedding_model,
        )

        self.embedder = SentenceTransformer(
            embedding_model
        )

        self.client = chromadb.PersistentClient(
            path=str(self.database_dir)
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        )

        LOGGER.info(
            "Knowledge database ready: %d chunks",
            self.count(),
        )

    def _iter_documents(
        self,
    ) -> Iterable[tuple[Path, str]]:
        """Discover supported documents."""

        for category in sorted(CATEGORIES):
            folder = self.knowledge_dir / category

            for path in sorted(folder.rglob("*")):
                if path.is_symlink():
                    LOGGER.warning(
                        "Skipping symbolic link: %s", path
                    )
                    continue

                if (
                    path.is_file()
                    and path.suffix.lower()
                    in SUPPORTED_EXTENSIONS
                ):
                    yield path, category

    @staticmethod
    def _extract_text(path: Path) -> str:
        """Extract text from a supported document."""

        extension = path.suffix.lower()

        if extension in {".txt", ".md"}:
            return path.read_text(
                encoding="utf-8-sig",
                errors="replace",
            )

        if extension == ".json":
            data = json.loads(
                path.read_text(encoding="utf-8-sig")
            )
            return json.dumps(
                data,
                ensure_ascii=False,
                indent=2,
            )

        if extension == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(str(path))

            return "\n\n".join(
                page.extract_text() or ""
                for page in reader.pages
            )

        raise KnowledgeRetrieverError(
            f"Unsupported document: {extension}"
        )

    def _split_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks."""

        text = re.sub(r"\r\n?", "\n", text)
        text = re.sub(r"[ \t]+", " ", text).strip()

        if not text:
            return []

        chunks = []
        start = 0

        while start < len(text):
            end = min(
                start + self.chunk_size,
                len(text),
            )

            if end < len(text):
                segment = text[start:end]

                boundaries = [
                    segment.rfind("\n\n"),
                    segment.rfind(". "),
                    segment.rfind("\n"),
                ]

                boundary = max(boundaries)

                if boundary >= self.chunk_size // 2:
                    end = start + boundary + 1

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end >= len(text):
                break

            start = max(
                start + 1,
                end - self.chunk_overlap,
            )

        return chunks

    def _embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Convert text into semantic embeddings."""

        vectors = self.embedder.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return vectors.tolist()

    def _source_key(self, path: Path) -> str:
        """Return a relative document path."""

        return path.relative_to(
            self.knowledge_dir
        ).as_posix()

    def _delete_source(self, source: str) -> int:
        """Delete indexed chunks for one source."""

        records = self.collection.get(
            where={"source": source}
        )

        ids = records.get("ids", [])

        if ids:
            self.collection.delete(ids=ids)

        return len(ids)

    def index_documents(
        self,
        *,
        remove_missing: bool = False,
    ) -> dict[str, int]:
        """Index documents and update changed files."""

        stats = {
            "indexed_files": 0,
            "skipped_files": 0,
            "failed_files": 0,
            "indexed_chunks": 0,
            "removed_chunks": 0,
        }

        seen = set()

        for path, category in self._iter_documents():
            source = self._source_key(path)
            seen.add(source)

            try:
                digest = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()

                existing = self.collection.get(
                    where={"source": source},
                    include=["metadatas"],
                )

                old_metadata = (
                    existing.get("metadatas") or []
                )

                if old_metadata and all(
                    item.get("sha256") == digest
                    for item in old_metadata
                ):
                    stats["skipped_files"] += 1
                    continue

                text = self._extract_text(path)
                chunks = self._split_text(text)

                if not chunks:
                    LOGGER.warning(
                        "No readable text: %s", source
                    )
                    stats["failed_files"] += 1
                    continue

                document_id = hashlib.sha256(
                    source.encode("utf-8")
                ).hexdigest()[:24]

                timestamp = datetime.now(
                    timezone.utc
                ).isoformat()

                ids = [
                    f"{document_id}:{digest[:12]}:{i:06d}"
                    for i in range(len(chunks))
                ]

                metadata = [
                    {
                        "source": source,
                        "category": category,
                        "chunk_index": i,
                        "sha256": digest,
                        "indexed_at": timestamp,
                        "document_id": document_id,
                    }
                    for i in range(len(chunks))
                ]

                vectors = []

                for start in range(
                    0, len(chunks), self.batch_size
                ):
                    batch = chunks[
                        start:start + self.batch_size
                    ]
                    vectors.extend(self._embed(batch))

                stats["removed_chunks"] += (
                    self._delete_source(source)
                )

                for start in range(
                    0, len(chunks), self.batch_size
                ):
                    end = start + self.batch_size

                    self.collection.upsert(
                        ids=ids[start:end],
                        documents=chunks[start:end],
                        metadatas=metadata[start:end],
                        embeddings=vectors[start:end],
                    )

                stats["indexed_files"] += 1
                stats["indexed_chunks"] += len(chunks)

                LOGGER.info(
                    "Indexed %s: %d chunks",
                    source,
                    len(chunks),
                )

            except Exception:
                stats["failed_files"] += 1
                LOGGER.exception(
                    "Indexing failed: %s", source
                )

        if remove_missing:
            stored = self.collection.get(
                include=["metadatas"]
            )

            stored_sources = {
                item["source"]
                for item in (
                    stored.get("metadatas") or []
                )
                if item and "source" in item
            }

            for source in stored_sources - seen:
                stats["removed_chunks"] += (
                    self._delete_source(source)
                )

        return stats

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        category: str | None = None,
        max_distance: float | None = None,
    ) -> list[RetrievalResult]:
        """Search indexed knowledge using semantic similarity."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                "Search query cannot be empty"
            )

        if top_k < 1:
            raise ValueError(
                "top_k must be positive"
            )

        if category is not None and category not in CATEGORIES:
            raise ValueError(
                f"Unknown knowledge category: {category}"
            )

        if max_distance is not None and max_distance < 0:
            raise ValueError(
                "max_distance cannot be negative"
            )

        total = self.count()

        if total == 0:
            LOGGER.warning(
                "Knowledge database is empty"
            )
            return []

        options = {
            "query_embeddings": self._embed(
                [query.strip()]
            ),
            "n_results": min(top_k, total),
            "include": [
                "documents",
                "metadatas",
                "distances",
            ],
        }

        if category:
            options["where"] = {
                "category": category
            }

        response = self.collection.query(**options)

        ids = response.get("ids", [[]])[0]
        texts = response.get("documents", [[]])[0]
        metadata = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]

        results = []

        for record_id, text, meta, distance in zip(
            ids, texts, metadata, distances
        ):
            if (
                max_distance is not None
                and distance > max_distance
            ):
                continue

            meta = meta or {}

            results.append(
                RetrievalResult(
                    text=text or "",
                    source=str(
                        meta.get("source", "")
                    ),
                    category=str(
                        meta.get("category", "")
                    ),
                    chunk_index=int(
                        meta.get("chunk_index", 0)
                    ),
                    distance=float(distance),
                    document_id=str(
                        meta.get(
                            "document_id",
                            record_id,
                        )
                    ),
                    metadata=dict(meta),
                )
            )

        return results

    def retrieve_context(
        self,
        query: str,
        *,
        top_k: int = 5,
        category: str | None = None,
        max_distance: float | None = None,
    ) -> str:
        """Return source-labelled context for an LLM."""

        matches = self.search(
            query,
            top_k=top_k,
            category=category,
            max_distance=max_distance,
        )

        if not matches:
            return (
                "No relevant documents were retrieved."
            )

        passages = []

        for match in matches:
            passages.append(
                f"[Source: {match.source} | "
                f"Chunk: {match.chunk_index} | "
                f"Distance: {match.distance:.4f}]\n"
                f"{match.text}"
            )

        return "\n\n".join(passages)

    def count(self) -> int:
        """Return the number of indexed chunks."""

        return self.collection.count()


def main() -> None:
    """Command-line interface for knowledge retrieval."""

    parser = argparse.ArgumentParser(
        description="AeroMind AI Knowledge Retriever"
    )

    parser.add_argument(
        "command",
        choices=["index", "search", "count"],
    )

    parser.add_argument(
        "query",
        nargs="?",
    )

    parser.add_argument(
        "--category",
        choices=sorted(CATEGORIES),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--max-distance",
        type=float,
    )

    parser.add_argument(
        "--remove-missing",
        action="store_true",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)-8s | "
            "%(name)s | %(message)s"
        ),
    )

    retriever = KnowledgeRetriever()

    if args.command == "index":
        results = retriever.index_documents(
            remove_missing=args.remove_missing
        )

        print(json.dumps(
            results,
            indent=2,
        ))

    elif args.command == "count":
        print(
            f"Indexed chunks: {retriever.count()}"
        )

    elif args.command == "search":
        if not args.query:
            parser.error(
                "The search command requires a query"
            )

        results = retriever.search(
            args.query,
            top_k=args.top_k,
            category=args.category,
            max_distance=args.max_distance,
        )

        print(json.dumps(
            [
                result.as_dict()
                for result in results
            ],
            indent=2,
            ensure_ascii=False,
        ))


if __name__ == "__main__":
    main()
