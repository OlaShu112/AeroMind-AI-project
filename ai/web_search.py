
"""
AeroMind AI - Public Web Search Engine
======================================

Module:
    ai.web_search

Purpose:
    Retrieve current public information using the Brave Search API
    and provide structured results to AeroMind AI's RAG pipeline.

Features:
    - Secure API key configuration
    - Query validation
    - Configurable search result limits
    - Request timeout handling
    - Retry with exponential backoff
    - HTTP and API error handling
    - URL validation
    - Duplicate result removal
    - Structured search metadata
    - Optional freshness and country filters
    - Safe logging without exposing credentials
    - Command-line interface
    - Compatible with existing RAG integrations

Security:
    Web search is read-only. Retrieved content is untrusted and
    must never be interpreted as executable flight instructions.

Dependencies:
    Python 3.12+
    Python standard library only

Environment:
    BRAVE_SEARCH_API_KEY=<your_api_key>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen


# ============================================================
# Logging
# ============================================================

logger = logging.getLogger("aeromind.web_search")
logger.addHandler(logging.NullHandler())


# ============================================================
# Exceptions
# ============================================================

class WebSearchError(Exception):
    """Base exception for web search failures."""


class WebSearchConfigurationError(WebSearchError):
    """Raised when API configuration is invalid."""


class WebSearchValidationError(WebSearchError):
    """Raised when a search request is invalid."""


class WebSearchAuthenticationError(WebSearchError):
    """Raised when Brave API authentication fails."""


class WebSearchRateLimitError(WebSearchError):
    """Raised when the API rate limit is exceeded."""


class WebSearchTimeoutError(WebSearchError):
    """Raised when a search request times out."""


class WebSearchAPIError(WebSearchError):
    """Raised when Brave returns an unexpected API error."""


class WebSearchResponseError(WebSearchError):
    """Raised when the API response is malformed."""


# ============================================================
# Data Models
# ============================================================

@dataclass(frozen=True)
class SearchResult:
    """
    Normalised public web search result.

    This structure is compatible with the AeroMind RAG
    knowledge retrieval pipeline.
    """

    title: str
    url: str
    snippet: str
    source_type: str = "web"
    provider: str = "brave"
    domain: str = ""
    published_at: Optional[str] = None
    retrieved_at: str = ""
    rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SearchResponse:
    """Complete structured search response."""

    query: str
    provider: str
    total_results: int
    results: list[SearchResult]
    retrieved_at: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["results"] = [
            result.to_dict()
            for result in self.results
        ]
        return data


# ============================================================
# Web Search Engine
# ============================================================

class WebSearch:
    """
    Brave Search API client for AeroMind AI.

    Example:
        engine = WebSearch()
        results = engine.search(
            "PX4 autonomous drone navigation",
            limit=5,
        )

    The search() method preserves the original interface
    and returns a list of dictionaries.
    """

    API_URL = (
        "https://api.search.brave.com/res/v1/web/search"
    )

    MAX_RESULTS = 20
    MAX_QUERY_LENGTH = 500

    ALLOWED_FRESHNESS = {
        "pd",  # Past day
        "pw",  # Past week
        "pm",  # Past month
        "py",  # Past year
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 15.0,
        max_retries: int = 2,
        retry_delay: float = 1.0,
        user_agent: str = "AeroMindAI/1.0",
    ) -> None:

        self.api_key = (
            api_key
            if api_key is not None
            else os.getenv("BRAVE_SEARCH_API_KEY", "")
        ).strip()

        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.user_agent = user_agent

        if not self.api_key:
            raise WebSearchConfigurationError(
                "BRAVE_SEARCH_API_KEY is not configured. "
                "Set the environment variable or provide "
                "api_key when initialising WebSearch."
            )

        if timeout <= 0:
            raise WebSearchConfigurationError(
                "Timeout must be greater than zero."
            )

        if max_retries < 0:
            raise WebSearchConfigurationError(
                "max_retries cannot be negative."
            )

        if retry_delay < 0:
            raise WebSearchConfigurationError(
                "retry_delay cannot be negative."
            )

        logger.info(
            "AeroMind web search engine initialised."
        )

    # --------------------------------------------------------
    # Input Validation
    # --------------------------------------------------------

    @staticmethod
    def _validate_query(query: str) -> str:

        if not isinstance(query, str):
            raise WebSearchValidationError(
                "Search query must be a string."
            )

        query = re.sub(r"\s+", " ", query).strip()

        if not query:
            raise WebSearchValidationError(
                "Search query cannot be empty."
            )

        if len(query) > WebSearch.MAX_QUERY_LENGTH:
            raise WebSearchValidationError(
                "Search query exceeds the maximum "
                f"length of {WebSearch.MAX_QUERY_LENGTH} "
                "characters."
            )

        if any(
            ord(character) < 32
            for character in query
        ):
            raise WebSearchValidationError(
                "Search query contains invalid characters."
            )

        return query

    @classmethod
    def _validate_limit(cls, limit: int) -> int:

        if isinstance(limit, bool) or not isinstance(
            limit, int
        ):
            raise WebSearchValidationError(
                "Result limit must be an integer."
            )

        if limit < 1:
            raise WebSearchValidationError(
                "Result limit must be at least 1."
            )

        return min(limit, cls.MAX_RESULTS)

    @classmethod
    def _validate_freshness(
        cls,
        freshness: Optional[str],
    ) -> Optional[str]:

        if freshness is None:
            return None

        if freshness not in cls.ALLOWED_FRESHNESS:
            raise WebSearchValidationError(
                "Invalid freshness value. "
                "Use pd, pw, pm or py."
            )

        return freshness

    @staticmethod
    def _validate_country(
        country: Optional[str],
    ) -> Optional[str]:

        if country is None:
            return None

        country = country.upper().strip()

        if not re.fullmatch(r"[A-Z]{2}", country):
            raise WebSearchValidationError(
                "Country must be a two-letter code."
            )

        return country

    # --------------------------------------------------------
    # URL Validation
    # --------------------------------------------------------

    @staticmethod
    def _is_valid_url(url: str) -> bool:
        """
        Accept only HTTP(S) search result URLs.

        This validates URL structure. It does not establish
        that a website or its content is trustworthy.
        """

        if not isinstance(url, str):
            return False

        if len(url) > 4096:
            return False

        if any(
            ord(character) < 33
            for character in url
        ):
            return False

        try:
            parsed = urlsplit(url)

            return (
                parsed.scheme.lower()
                in {"http", "https"}
                and bool(parsed.hostname)
                and parsed.username is None
                and parsed.password is None
            )

        except ValueError:
            return False

    @staticmethod
    def _extract_domain(url: str) -> str:

        try:
            return (
                urlsplit(url).hostname or ""
            ).lower()

        except ValueError:
            return ""

    # --------------------------------------------------------
    # Request Construction
    # --------------------------------------------------------

    def _build_url(
        self,
        query: str,
        limit: int,
        freshness: Optional[str] = None,
        country: Optional[str] = None,
    ) -> str:

        parameters = {
            "q": query,
            "count": limit,
        }

        if freshness:
            parameters["freshness"] = freshness

        if country:
            parameters["country"] = country

        return (
            f"{self.API_URL}?"
            f"{urlencode(parameters)}"
        )

    def _build_request(self, url: str) -> Request:

        return Request(
            url,
            headers={
                "X-Subscription-Token": self.api_key,
                "Accept": "application/json",
                "User-Agent": self.user_agent,
            },
            method="GET",
        )

    # --------------------------------------------------------
    # Retry Handling
    # --------------------------------------------------------

    def _retry_wait(
        self,
        attempt: int,
        retry_after: Optional[str] = None,
    ) -> None:

        if retry_after:
            try:
                delay = float(retry_after)
                delay = max(0.0, min(delay, 60.0))
            except (TypeError, ValueError):
                delay = None
        else:
            delay = None

        if delay is None:
            delay = min(
                self.retry_delay * (2 ** attempt),
                30.0,
            )
            delay += random.uniform(0, 0.25)

        logger.warning(
            "Retrying web search after %.2f seconds.",
            delay,
        )

        time.sleep(delay)

    # --------------------------------------------------------
    # HTTP Communication
    # --------------------------------------------------------

    def _request_json(
        self,
        request: Request,
    ) -> dict[str, Any]:

        for attempt in range(self.max_retries + 1):

            try:
                with urlopen(
                    request,
                    timeout=self.timeout,
                ) as response:

                    status = response.status

                    if status != 200:
                        raise WebSearchAPIError(
                            f"Unexpected HTTP status: {status}"
                        )

                    # Limit response size to 5 MiB.
                    # Avoid loading arbitrarily large data.
                    maximum_bytes = 5 * 1024 * 1024

                    payload = response.read(
                        maximum_bytes + 1
                    )

                    if len(payload) > maximum_bytes:
                        raise WebSearchResponseError(
                            "API response exceeds "
                            "the maximum permitted size."
                        )

                    try:
                        data = json.loads(
                            payload.decode("utf-8")
                        )

                    except (
                        UnicodeDecodeError,
                        json.JSONDecodeError,
                    ) as exc:

                        raise WebSearchResponseError(
                            "Brave returned invalid JSON."
                        ) from exc

                    if not isinstance(data, dict):
                        raise WebSearchResponseError(
                            "Unexpected API response format."
                        )

                    return data

            except HTTPError as exc:

                status = exc.code

                if status in (401, 403):
                    raise WebSearchAuthenticationError(
                        "Brave Search authentication "
                        "failed or access was denied. "
                        "Check the API key and subscription."
                    ) from exc

                if status == 429:

                    if attempt < self.max_retries:
                        self._retry_wait(
                            attempt,
                            exc.headers.get(
                                "Retry-After"
                            ),
                        )
                        continue

                    raise WebSearchRateLimitError(
                        "Brave Search API rate limit "
                        "exceeded. Try again later."
                    ) from exc

                if (
                    status in (408, 500, 502, 503, 504)
                    and attempt < self.max_retries
                ):
                    self._retry_wait(attempt)
                    continue

                raise WebSearchAPIError(
                    f"Brave Search API returned "
                    f"HTTP {status}."
                ) from exc

            except (
                TimeoutError,
                URLError,
            ) as exc:

                if attempt < self.max_retries:
                    self._retry_wait(attempt)
                    continue

                raise WebSearchTimeoutError(
                    "Unable to reach Brave Search "
                    "within the configured timeout "
                    "or due to a network failure."
                ) from exc

        raise WebSearchAPIError(
            "Web search failed after all retries."
        )

    # --------------------------------------------------------
    # Result Normalisation
    # --------------------------------------------------------

    @staticmethod
    def _clean_text(
        value: Any,
        max_length: int = 2000,
    ) -> str:

        if not isinstance(value, str):
            return ""

        # Remove HTML tags and normalise whitespace.
        # This is for plain-text display, not HTML
        # sanitisation or safe HTML rendering.
        cleaned = re.sub(
            r"<[^>]*>",
            " ",
            value,
        )

        cleaned = re.sub(
            r"\s+",
            " ",
            cleaned,
        ).strip()

        return cleaned[:max_length]

    def _normalise_results(
        self,
        data: dict[str, Any],
        limit: int,
    ) -> list[SearchResult]:

        web_data = data.get("web", {})

        if not isinstance(web_data, dict):
            raise WebSearchResponseError(
                "Invalid web results structure."
            )

        raw_results = web_data.get("results", [])

        if not isinstance(raw_results, list):
            raise WebSearchResponseError(
                "Web results must be a list."
            )

        results: list[SearchResult] = []
        seen_urls: set[str] = set()

        retrieved_at = datetime.now(
            timezone.utc
        ).isoformat()

        for item in raw_results:

            if not isinstance(item, dict):
                continue

            url = item.get("url", "")

            if not self._is_valid_url(url):
                continue

            # Remove URL fragments for deduplication.
            parsed = urlsplit(url)
            canonical_url = parsed._replace(
                fragment=""
            ).geturl()

            if canonical_url in seen_urls:
                continue

            seen_urls.add(canonical_url)

            title = self._clean_text(
                item.get("title", ""),
                max_length=300,
            )

            snippet = self._clean_text(
                item.get("description", ""),
            )

            if not title:
                title = self._extract_domain(url)

            published_at = item.get(
                "page_age"
            )

            if not isinstance(
                published_at, str
            ):
                published_at = None

            result = SearchResult(
                title=title,
                url=url,
                snippet=snippet,
                domain=self._extract_domain(url),
                published_at=published_at,
                retrieved_at=retrieved_at,
                rank=len(results) + 1,
            )

            results.append(result)

            if len(results) >= limit:
                break

        return results

    # --------------------------------------------------------
    # Public Search API
    # --------------------------------------------------------

    def search(
        self,
        query: str,
        limit: int = 5,
        freshness: Optional[str] = None,
        country: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Search the public web.

        Args:
            query:
                User's search query.

            limit:
                Number of results, between 1 and 20.

            freshness:
                Optional freshness filter:
                    pd = past day
                    pw = past week
                    pm = past month
                    py = past year

            country:
                Optional two-letter country code,
                such as GB or US.

        Returns:
            List of structured result dictionaries.

        Raises:
            WebSearchError:
                For validation, network or API failures.
        """

        response = self.search_detailed(
            query=query,
            limit=limit,
            freshness=freshness,
            country=country,
        )

        return [
            result.to_dict()
            for result in response.results
        ]

    def search_detailed(
        self,
        query: str,
        limit: int = 5,
        freshness: Optional[str] = None,
        country: Optional[str] = None,
    ) -> SearchResponse:
        """
        Search and return structured response metadata.
        """

        query = self._validate_query(query)
        limit = self._validate_limit(limit)

        freshness = self._validate_freshness(
            freshness
        )

        country = self._validate_country(
            country
        )

        logger.info(
            "Starting Brave web search: %s",
            query,
        )

        url = self._build_url(
            query=query,
            limit=limit,
            freshness=freshness,
            country=country,
        )

        request = self._build_request(url)

        data = self._request_json(request)

        results = self._normalise_results(
            data,
            limit,
        )

        logger.info(
            "Web search completed with %d results.",
            len(results),
        )

        return SearchResponse(
            query=query,
            provider="brave",
            total_results=len(results),
            results=results,
            retrieved_at=datetime.now(
                timezone.utc
            ).isoformat(),
        )

    # --------------------------------------------------------
    # RAG Context Preparation
    # --------------------------------------------------------

    def search_for_rag(
        self,
        query: str,
        limit: int = 5,
        max_snippet_chars: int = 1200,
    ) -> dict[str, Any]:
        """
        Retrieve search results and prepare context
        for the AeroMind AI RAG engine.

        IMPORTANT:
            Search snippets are untrusted source content.
            They must not override system instructions,
            safety checks or operator authorisation.
        """

        if max_snippet_chars < 1:
            raise WebSearchValidationError(
                "max_snippet_chars must be positive."
            )

        response = self.search_detailed(
            query=query,
            limit=limit,
        )

        context_parts = []
        sources = []

        for result in response.results:

            snippet = result.snippet[
                :max_snippet_chars
            ]

            context_parts.append(
                f"[Source {result.rank}]\n"
                f"Title: {result.title}\n"
                f"URL: {result.url}\n"
                f"Excerpt: {snippet}"
            )

            sources.append({
                "id": result.rank,
                "title": result.title,
                "url": result.url,
                "domain": result.domain,
                "source_type": "web",
                "provider": "brave",
            })

        return {
            "query": response.query,
            "context": "\n\n".join(context_parts),
            "sources": sources,
            "result_count": len(sources),
            "retrieved_at": response.retrieved_at,
            "trust_level": "untrusted_external",
        }


# ============================================================
# Command-Line Interface
# ============================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "AeroMind AI - Brave Web Search Engine"
        )
    )

    parser.add_argument(
        "query",
        nargs="+",
        help="Search query",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum results (1-20)",
    )

    parser.add_argument(
        "--freshness",
        choices=["pd", "pw", "pm", "py"],
        default=None,
        help="Filter results by freshness",
    )

    parser.add_argument(
        "--country",
        default=None,
        help="Two-letter country code",
    )

    parser.add_argument(
        "--rag",
        action="store_true",
        help="Return RAG-ready context",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable diagnostic logging",
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

    query = " ".join(args.query)

    try:
        engine = WebSearch()

        if args.rag:
            output = engine.search_for_rag(
                query=query,
                limit=args.limit,
            )

        else:
            response = engine.search_detailed(
                query=query,
                limit=args.limit,
                freshness=args.freshness,
                country=args.country,
            )

            output = response.to_dict()

        print(
            json.dumps(
                output,
                indent=2,
                ensure_ascii=False,
            )
        )

        return 0

    except WebSearchError as exc:

        logger.error(
            "Web search failed: %s",
            exc,
        )

        print(
            json.dumps({
                "success": False,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }),
            file=sys.stderr,
        )

        return 1

    except KeyboardInterrupt:

        logger.warning(
            "Web search interrupted by operator."
        )

        return 130


if __name__ == "__main__":
    sys.exit(main())
