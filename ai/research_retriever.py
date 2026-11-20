
"""
AeroMind AI - Research and Encyclopedia Retriever
==================================================

Module:
    ai.research_retriever

Purpose:
    Retrieve academic research and encyclopedia information
    for the AeroMind AI Retrieval-Augmented Generation
    (RAG) knowledge system.

Data Sources:
    1. OpenAlex - Academic research publications
    2. Wikipedia - Encyclopedic knowledge

Features:
    - Academic research search
    - Wikipedia encyclopedia search
    - Publication metadata extraction
    - Research abstract reconstruction
    - DOI and publication URL handling
    - Author and citation information
    - Search query validation
    - Request retry and timeout handling
    - API rate-limit handling
    - Duplicate result filtering
    - Structured RAG-ready results
    - Combined multi-source search
    - Command-line interface
    - Safe logging

Security:
    Retrieved content is untrusted external information.
    It must not override AeroMind AI safety controls
    or directly execute drone flight commands.

Dependencies:
    requests
    beautifulsoup4

Python:
    3.12+
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import quote, urlsplit

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger(
    "aeromind.research_retriever"
)

logger.addHandler(logging.NullHandler())


# ============================================================
# EXCEPTIONS
# ============================================================

class ResearchRetrieverError(Exception):
    """Base exception for research retrieval failures."""


class ResearchValidationError(ResearchRetrieverError):
    """Raised when input parameters are invalid."""


class ResearchAPIError(ResearchRetrieverError):
    """Raised when an external research API fails."""


class ResearchTimeoutError(ResearchRetrieverError):
    """Raised when an API request times out."""


class ResearchRateLimitError(ResearchRetrieverError):
    """Raised when an API rate limit is exceeded."""


class ResearchResponseError(ResearchRetrieverError):
    """Raised when an API response is malformed."""


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class ResearchConfig:
    """Configuration for academic and encyclopedia search."""

    timeout: float = 20.0
    max_results: int = 10
    max_query_length: int = 500
    max_response_bytes: int = 5_000_000
    max_retries: int = 2
    backoff_factor: float = 0.5

    user_agent: str = (
        "AeroMindAI/1.0 "
        "(research knowledge assistant)"
    )

    openalex_url: str = (
        "https://api.openalex.org/works"
    )

    wikipedia_url: str = (
        "https://en.wikipedia.org/w/api.php"
    )

    def __post_init__(self) -> None:

        if self.timeout <= 0:
            raise ValueError(
                "Timeout must be greater than zero."
            )

        if self.max_results < 1:
            raise ValueError(
                "max_results must be positive."
            )

        if self.max_query_length < 1:
            raise ValueError(
                "max_query_length must be positive."
            )

        if self.max_response_bytes < 1:
            raise ValueError(
                "max_response_bytes must be positive."
            )

        if self.max_retries < 0:
            raise ValueError(
                "max_retries cannot be negative."
            )

        if self.backoff_factor < 0:
            raise ValueError(
                "backoff_factor cannot be negative."
            )


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class ResearchResult:
    """
    Normalised research or encyclopedia result.

    Core fields match AeroMind AI's web search interface.
    """

    title: str
    url: str
    snippet: str
    source_type: str

    provider: str = ""
    authors: tuple[str, ...] = ()
    publication_year: Optional[int] = None
    doi: Optional[str] = None
    cited_by_count: Optional[int] = None
    abstract: str = ""
    retrieved_at: str = ""
    trust_level: str = "untrusted_external"

    def to_dict(self) -> dict[str, Any]:

        data = asdict(self)

        data["authors"] = list(self.authors)

        return data


@dataclass(frozen=True)
class CombinedResearchResponse:
    """Combined multi-source research response."""

    query: str
    total_results: int
    results: list[ResearchResult]
    retrieved_at: str

    def to_dict(self) -> dict[str, Any]:

        return {
            "query": self.query,
            "total_results": self.total_results,
            "results": [
                item.to_dict()
                for item in self.results
            ],
            "retrieved_at": self.retrieved_at,
        }


# ============================================================
# HTML TEXT CLEANING
# ============================================================

class BeautifulText:
    """
    Safely convert HTML fragments into plain text.

    The output is text only and must still be treated
    as untrusted external content.
    """

    @staticmethod
    def clean(value: Any) -> str:

        if not isinstance(value, str):
            return ""

        soup = BeautifulSoup(
            value,
            "html.parser",
        )

        for tag in soup(
            ["script", "style", "noscript"]
        ):
            tag.decompose()

        text = soup.get_text(
            " ",
            strip=True,
        )

        return re.sub(
            r"\s+",
            " ",
            text,
        ).strip()


# ============================================================
# RESEARCH RETRIEVER
# ============================================================

class ResearchRetriever:
    """
    Public research and encyclopedia retrieval engine.

    Supported sources:
        OpenAlex
        Wikipedia

    Example:
        retriever = ResearchRetriever()

        papers = retriever.search_openalex(
            "autonomous drone navigation",
            limit=5,
        )

        articles = retriever.search_wikipedia(
            "unmanned aerial vehicle",
            limit=3,
        )
    """

    def __init__(
        self,
        timeout: float = 20.0,
        max_retries: int = 2,
        openalex_api_key: Optional[str] = None,
        contact_email: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> None:

        self.config = ResearchConfig(
            timeout=timeout,
            max_retries=max_retries,
        )

        self.openalex_api_key = (
            openalex_api_key
            if openalex_api_key is not None
            else os.getenv("OPENALEX_API_KEY", "")
        ).strip()

        self.contact_email = (
            contact_email
            if contact_email is not None
            else os.getenv("AEROMIND_CONTACT_EMAIL", "")
        ).strip()

        self.user_agent = (
            user_agent
            or self.config.user_agent
        )

        self.session = self._create_session()

        logger.info(
            "AeroMind ResearchRetriever initialised."
        )

    # --------------------------------------------------------
    # HTTP SESSION
    # --------------------------------------------------------

    def _create_session(
        self,
    ) -> requests.Session:
        """Create an HTTP session with retry handling."""

        session = requests.Session()

        retries = Retry(
            total=self.config.max_retries,
            connect=self.config.max_retries,
            read=self.config.max_retries,
            status=self.config.max_retries,
            backoff_factor=(
                self.config.backoff_factor
            ),
            status_forcelist=[
                429,
                500,
                502,
                503,
                504,
            ],
            allowed_methods={"GET"},
            respect_retry_after_header=True,
            raise_on_status=False,
        )

        adapter = HTTPAdapter(
            max_retries=retries
        )

        session.mount(
            "https://",
            adapter,
        )

        session.headers.update({
            "User-Agent": self.user_agent,
            "Accept": "application/json",
        })

        return session

    # --------------------------------------------------------
    # INPUT VALIDATION
    # --------------------------------------------------------

    def _validate_query(
        self,
        query: str,
    ) -> str:

        if not isinstance(query, str):
            raise ResearchValidationError(
                "Search query must be a string."
            )

        query = re.sub(
            r"\s+",
            " ",
            query,
        ).strip()

        if not query:
            raise ResearchValidationError(
                "Search query cannot be empty."
            )

        if len(query) > self.config.max_query_length:
            raise ResearchValidationError(
                "Search query exceeds the permitted length."
            )

        if any(
            ord(character) < 32
            for character in query
        ):
            raise ResearchValidationError(
                "Search query contains control characters."
            )

        return query

    def _validate_limit(
        self,
        limit: int,
    ) -> int:

        if isinstance(limit, bool) or not isinstance(
            limit, int
        ):
            raise ResearchValidationError(
                "Result limit must be an integer."
            )

        if limit < 1:
            raise ResearchValidationError(
                "Result limit must be positive."
            )

        return min(
            limit,
            self.config.max_results,
        )

    # --------------------------------------------------------
    # TEXT UTILITIES
    # --------------------------------------------------------

    @staticmethod
    def _clean_text(
        value: Any,
        max_length: int = 3000,
    ) -> str:

        if not isinstance(value, str):
            return ""

        text = BeautifulText.clean(value)

        return text[:max_length]

    @staticmethod
    def _current_timestamp() -> str:

        return datetime.now(
            timezone.utc
        ).isoformat()

    @staticmethod
    def _safe_url(value: Any) -> str:
        """
        Accept only valid HTTP(S) reference URLs.

        This does not authorise fetching the URL.
        """

        if not isinstance(value, str):
            return ""

        if len(value) > 4096:
            return ""

        try:
            parsed = urlsplit(value)

            if (
                parsed.scheme.lower()
                not in {"http", "https"}
            ):
                return ""

            if not parsed.hostname:
                return ""

            if parsed.username or parsed.password:
                return ""

            return value

        except ValueError:
            return ""

    # --------------------------------------------------------
    # API REQUEST HANDLING
    # --------------------------------------------------------

    def _request_json(
        self,
        url: str,
        params: dict[str, Any],
        headers: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Send a bounded JSON API request."""

        response = None

        try:
            response = self.session.get(
                url,
                params=params,
                headers=headers,
                timeout=self.config.timeout,
                stream=True,
                allow_redirects=False,
            )

            if 300 <= response.status_code < 400:
                raise ResearchAPIError(
                    "Unexpected API redirect."
                )

            if response.status_code == 429:
                raise ResearchRateLimitError(
                    "Research API rate limit exceeded."
                )

            if response.status_code >= 400:
                raise ResearchAPIError(
                    "Research API returned HTTP "
                    f"{response.status_code}."
                )

            declared_size = response.headers.get(
                "Content-Length"
            )

            if declared_size:
                try:
                    size = int(declared_size)

                    if (
                        size
                        > self.config.max_response_bytes
                    ):
                        raise ResearchResponseError(
                            "API response exceeds size limit."
                        )

                except ValueError:
                    pass

            chunks = []
            total_bytes = 0

            for chunk in response.iter_content(
                chunk_size=8192
            ):

                if not chunk:
                    continue

                total_bytes += len(chunk)

                if (
                    total_bytes
                    > self.config.max_response_bytes
                ):
                    raise ResearchResponseError(
                        "API response exceeds size limit."
                    )

                chunks.append(chunk)

            try:
                data = json.loads(
                    b"".join(chunks).decode("utf-8")
                )

            except (
                UnicodeDecodeError,
                json.JSONDecodeError,
            ) as exc:

                raise ResearchResponseError(
                    "API returned invalid JSON."
                ) from exc

            if not isinstance(data, dict):
                raise ResearchResponseError(
                    "Unexpected API response structure."
                )

            return data

        except requests.Timeout as exc:
            raise ResearchTimeoutError(
                "Research API request timed out."
            ) from exc

        except requests.RequestException as exc:
            raise ResearchAPIError(
                "Unable to contact research API."
            ) from exc

        finally:
            if response is not None:
                response.close()

    # --------------------------------------------------------
    # OPENALEX ABSTRACT RECONSTRUCTION
    # --------------------------------------------------------

    @staticmethod
    def _reconstruct_abstract(
        inverted_index: Any,
    ) -> str:
        """
        Reconstruct abstract text from the OpenAlex
        abstract_inverted_index representation.
        """

        if not isinstance(
            inverted_index,
            dict,
        ):
            return ""

        positions = {}

        for word, indices in inverted_index.items():

            if not isinstance(word, str):
                continue

            if not isinstance(indices, list):
                continue

            for position in indices:

                if (
                    isinstance(position, int)
                    and not isinstance(position, bool)
                    and 0 <= position <= 100_000
                ):
                    positions[position] = word

        if not positions:
            return ""

        max_position = max(positions)

        if max_position > 10_000:
            return ""

        words = [
            positions.get(index, "")
            for index in range(max_position + 1)
        ]

        return " ".join(words).strip()

    # --------------------------------------------------------
    # OPENALEX AUTHORS
    # --------------------------------------------------------

    @staticmethod
    def _extract_authors(
        authorships: Any,
        max_authors: int = 10,
    ) -> tuple[str, ...]:

        if not isinstance(authorships, list):
            return ()

        authors = []

        for authorship in authorships:

            if not isinstance(
                authorship,
                dict,
            ):
                continue

            author = authorship.get(
                "author",
                {},
            )

            if not isinstance(author, dict):
                continue

            name = author.get(
                "display_name",
                "",
            )

            if isinstance(name, str) and name.strip():
                authors.append(name.strip())

            if len(authors) >= max_authors:
                break

        return tuple(authors)

    # --------------------------------------------------------
    # OPENALEX SEARCH
    # --------------------------------------------------------

    def search_openalex(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Search academic publications through OpenAlex.

        Returns:
            List of structured research results.
        """

        query = self._validate_query(query)
        limit = self._validate_limit(limit)

        params = {
            "search": query,
            "per-page": limit,
        }

        if self.openalex_api_key:
            params["api_key"] = (
                self.openalex_api_key
            )

        if self.contact_email:
            params["mailto"] = (
                self.contact_email
            )

        logger.info(
            "Searching OpenAlex for research."
        )

        data = self._request_json(
            self.config.openalex_url,
            params=params,
        )

        raw_results = data.get(
            "results",
            [],
        )

        if not isinstance(raw_results, list):
            raise ResearchResponseError(
                "Invalid OpenAlex results structure."
            )

        results = []
        seen_urls = set()

        for item in raw_results:

            if not isinstance(item, dict):
                continue

            title = self._clean_text(
                item.get("title", ""),
                max_length=500,
            )

            location = (
                item.get("best_oa_location")
                or {}
            )

            if not isinstance(location, dict):
                location = {}

            candidate_urls = [
                location.get("landing_page_url"),
                item.get("doi"),
                item.get("id"),
            ]

            url = ""

            for candidate in candidate_urls:

                validated = self._safe_url(
                    candidate
                )

                if validated:
                    url = validated
                    break

            if not title or not url:
                continue

            if url in seen_urls:
                continue

            seen_urls.add(url)

            year = item.get(
                "publication_year"
            )

            if (
                not isinstance(year, int)
                or isinstance(year, bool)
            ):
                year = None

            citation_count = item.get(
                "cited_by_count"
            )

            if (
                not isinstance(
                    citation_count,
                    int,
                )
                or isinstance(
                    citation_count,
                    bool,
                )
            ):
                citation_count = None

            abstract = self._reconstruct_abstract(
                item.get(
                    "abstract_inverted_index"
                )
            )

            abstract = self._clean_text(
                abstract,
                max_length=5000,
            )

            authors = self._extract_authors(
                item.get(
                    "authorships",
                    [],
                )
            )

            doi = self._safe_url(
                item.get("doi")
            ) or None

            snippet = abstract[:1000]

            if not snippet:
                snippet = (
                    "Academic research publication. "
                    f"Year: {year or 'unknown'}. "
                    f"Citations: "
                    f"{citation_count if citation_count is not None else 'unknown'}."
                )

            result = ResearchResult(
                title=title,
                url=url,
                snippet=snippet,
                source_type="research",
                provider="openalex",
                authors=authors,
                publication_year=year,
                doi=doi,
                cited_by_count=citation_count,
                abstract=abstract,
                retrieved_at=(
                    self._current_timestamp()
                ),
            )

            results.append(
                result.to_dict()
            )

        return results

    # --------------------------------------------------------
    # WIKIPEDIA SEARCH
    # --------------------------------------------------------

    def search_wikipedia(
        self,
        query: str,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Search English Wikipedia articles.

        Returns:
            List of encyclopedia results.
        """

        query = self._validate_query(query)
        limit = self._validate_limit(limit)

        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "format": "json",
            "formatversion": 2,
        }

        logger.info(
            "Searching Wikipedia encyclopedia."
        )

        data = self._request_json(
            self.config.wikipedia_url,
            params=params,
        )

        query_data = data.get(
            "query",
            {},
        )

        if not isinstance(query_data, dict):
            raise ResearchResponseError(
                "Invalid Wikipedia response structure."
            )

        raw_results = query_data.get(
            "search",
            [],
        )

        if not isinstance(raw_results, list):
            raise ResearchResponseError(
                "Invalid Wikipedia search results."
            )

        results = []
        seen_titles = set()

        for item in raw_results:

            if not isinstance(item, dict):
                continue

            title = self._clean_text(
                item.get("title", ""),
                max_length=500,
            )

            if not title:
                continue

            if title.lower() in seen_titles:
                continue

            seen_titles.add(
                title.lower()
            )

            encoded_title = quote(
                title.replace(" ", "_"),
                safe="",
            )

            url = (
                "https://en.wikipedia.org/wiki/"
                f"{encoded_title}"
            )

            snippet = BeautifulText.clean(
                item.get("snippet", "")
            )

            snippet = snippet[:1500]

            result = ResearchResult(
                title=title,
                url=url,
                snippet=snippet,
                source_type="encyclopedia",
                provider="wikipedia",
                retrieved_at=(
                    self._current_timestamp()
                ),
            )

            results.append(
                result.to_dict()
            )

        return results

    # --------------------------------------------------------
    # COMBINED RESEARCH SEARCH
    # --------------------------------------------------------

    def search_all(
        self,
        query: str,
        research_limit: int = 5,
        wikipedia_limit: int = 3,
        continue_on_error: bool = True,
    ) -> dict[str, Any]:
        """
        Search OpenAlex and Wikipedia.

        If continue_on_error=True, one provider failure
        does not prevent results from the other provider.
        """

        query = self._validate_query(query)

        results = []
        errors = []

        try:
            results.extend(
                self.search_openalex(
                    query,
                    limit=research_limit,
                )
            )

        except ResearchRetrieverError as exc:

            if not continue_on_error:
                raise

            logger.warning(
                "OpenAlex search failed: %s",
                exc,
            )

            errors.append({
                "provider": "openalex",
                "error": str(exc),
            })

        try:
            results.extend(
                self.search_wikipedia(
                    query,
                    limit=wikipedia_limit,
                )
            )

        except ResearchRetrieverError as exc:

            if not continue_on_error:
                raise

            logger.warning(
                "Wikipedia search failed: %s",
                exc,
            )

            errors.append({
                "provider": "wikipedia",
                "error": str(exc),
            })

        unique_results = []
        seen_urls = set()

        for result in results:

            url = result.get(
                "url",
                "",
            )

            if not url or url in seen_urls:
                continue

            seen_urls.add(url)

            unique_results.append(result)

        return {
            "query": query,
            "total_results": len(unique_results),
            "results": unique_results,
            "errors": errors,
            "retrieved_at": (
                self._current_timestamp()
            ),
        }

    # --------------------------------------------------------
    # RAG CONTEXT GENERATION
    # --------------------------------------------------------

    def search_for_rag(
        self,
        query: str,
        research_limit: int = 5,
        wikipedia_limit: int = 3,
        max_snippet_chars: int = 1500,
    ) -> dict[str, Any]:
        """
        Search both providers and prepare context
        for the AeroMind AI RAG engine.

        Research snippets and abstracts are untrusted.
        """

        if max_snippet_chars < 1:
            raise ResearchValidationError(
                "max_snippet_chars must be positive."
            )

        response = self.search_all(
            query=query,
            research_limit=research_limit,
            wikipedia_limit=wikipedia_limit,
        )

        context_parts = []
        sources = []

        for index, result in enumerate(
            response["results"],
            start=1,
        ):

            snippet = result.get(
                "snippet",
                "",
            )[:max_snippet_chars]

            context_parts.append(
                f"[Source {index}]\n"
                f"Title: {result['title']}\n"
                f"Provider: {result['provider']}\n"
                f"URL: {result['url']}\n"
                f"Content: {snippet}"
            )

            sources.append({
                "id": index,
                "title": result["title"],
                "url": result["url"],
                "source_type": (
                    result["source_type"]
                ),
                "provider": result["provider"],
                "publication_year": (
                    result.get(
                        "publication_year"
                    )
                ),
            })

        return {
            "query": response["query"],
            "context": "\n\n".join(
                context_parts
            ),
            "sources": sources,
            "result_count": len(sources),
            "errors": response["errors"],
            "retrieved_at": (
                response["retrieved_at"]
            ),
            "trust_level": (
                "untrusted_external"
            ),
        }

    # --------------------------------------------------------
    # RESOURCE MANAGEMENT
    # --------------------------------------------------------

    def close(self) -> None:
        """Close the HTTP session."""

        self.session.close()

    def __enter__(self) -> "ResearchRetriever":
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:

        self.close()


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

def main() -> int:
    """Run research searches from the terminal."""

    parser = argparse.ArgumentParser(
        description=(
            "AeroMind AI - Research and "
            "Encyclopedia Retriever"
        )
    )

    parser.add_argument(
        "query",
        nargs="+",
        help="Research search query.",
    )

    parser.add_argument(
        "--source",
        choices=[
            "all",
            "openalex",
            "wikipedia",
        ],
        default="all",
        help="Select research provider.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum results per selected provider.",
    )

    parser.add_argument(
        "--rag",
        action="store_true",
        help="Return RAG-ready context.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable diagnostic logging.",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format=(
                "%(asctime)s | %(levelname)s | "
                "%(name)s | %(message)s"
            ),
            stream=sys.stderr,
        )

    query = " ".join(
        args.query
    )

    try:
        with ResearchRetriever() as retriever:

            if args.source == "openalex":

                results = retriever.search_openalex(
                    query,
                    limit=args.limit,
                )

                output = {
                    "query": query,
                    "source": "openalex",
                    "results": results,
                    "total_results": len(results),
                }

            elif args.source == "wikipedia":

                results = retriever.search_wikipedia(
                    query,
                    limit=args.limit,
                )

                output = {
                    "query": query,
                    "source": "wikipedia",
                    "results": results,
                    "total_results": len(results),
                }

            elif args.rag:

                output = retriever.search_for_rag(
                    query,
                    research_limit=args.limit,
                    wikipedia_limit=args.limit,
                )

            else:

                output = retriever.search_all(
                    query,
                    research_limit=args.limit,
                    wikipedia_limit=args.limit,
                )

        print(
            json.dumps(
                output,
                indent=2,
                ensure_ascii=False,
            )
        )

        return 0

    except ResearchRetrieverError as exc:

        print(
            json.dumps({
                "success": False,
                "error": str(exc),
                "error_type": (
                    type(exc).__name__
                ),
            }),
            file=sys.stderr,
        )

        return 1

    except KeyboardInterrupt:

        logger.warning(
            "Research retrieval interrupted."
        )

        return 130


if __name__ == "__main__":
    sys.exit(main())
