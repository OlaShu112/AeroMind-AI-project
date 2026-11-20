
"""
AeroMind AI - Secure Public Web Retriever
==========================================

Module:
    ai.web_retriever

Description:
    Securely retrieves publicly accessible HTTPS web pages
    and extracts readable text for the AeroMind AI
    Retrieval-Augmented Generation (RAG) pipeline.

Features:
    - HTTPS-only connections
    - Public IP address validation
    - DNS rebinding protection through IP pinning
    - TLS certificate and hostname verification
    - Redirect blocking
    - Proxy bypass
    - Connection and read timeouts
    - Maximum response-size enforcement
    - HTML content-type validation
    - HTML parsing and text extraction
    - Metadata extraction
    - Duplicate whitespace removal
    - Structured RAG-ready output
    - Detailed error handling
    - Command-line interface

Security:
    Retrieved web content is untrusted external information.
    It must never directly authorise drone flight commands,
    modify safety policies or execute system instructions.

Dependencies:
    requests
    beautifulsoup4
    urllib3

Python:
    3.12+
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import logging
import re
import socket
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3 import PoolManager
from urllib3.connection import HTTPSConnection
from urllib3.connectionpool import HTTPSConnectionPool


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger("aeromind.web_retriever")
logger.addHandler(logging.NullHandler())


# ============================================================
# EXCEPTIONS
# ============================================================

class RetrievalError(Exception):
    """Base exception for retrieval failures."""


class URLValidationError(RetrievalError):
    """Raised when a URL fails security validation."""


class DNSResolutionError(RetrievalError):
    """Raised when DNS resolution fails or is unsafe."""


class RetrievalTimeoutError(RetrievalError):
    """Raised when a retrieval request times out."""


class RetrievalHTTPError(RetrievalError):
    """Raised when the remote server returns an HTTP error."""


class RetrievalContentError(RetrievalError):
    """Raised when retrieved content is unsupported or invalid."""


class RetrievalSizeError(RetrievalError):
    """Raised when a response exceeds its configured size limit."""


class RetrievalConnectionError(RetrievalError):
    """Raised when a secure connection cannot be established."""


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class RetrieverConfig:
    """Configuration for restricted web retrieval."""

    max_chars: int = 8000
    max_bytes: int = 1_000_000
    connect_timeout: float = 5.0
    read_timeout: float = 15.0
    max_url_length: int = 2048
    max_dns_addresses: int = 16
    user_agent: str = "AeroMindAI/1.0"

    def __post_init__(self) -> None:

        if self.max_chars < 1:
            raise ValueError(
                "max_chars must be greater than zero."
            )

        if self.max_bytes < 1:
            raise ValueError(
                "max_bytes must be greater than zero."
            )

        if self.connect_timeout <= 0:
            raise ValueError(
                "connect_timeout must be positive."
            )

        if self.read_timeout <= 0:
            raise ValueError(
                "read_timeout must be positive."
            )

        if self.max_dns_addresses < 1:
            raise ValueError(
                "max_dns_addresses must be positive."
            )


# ============================================================
# RETRIEVED DOCUMENT MODEL
# ============================================================

@dataclass(frozen=True)
class RetrievedDocument:
    """Normalised document for the RAG pipeline."""

    title: str
    url: str
    text: str
    source_type: str
    domain: str
    retrieved_at: str
    content_type: str
    character_count: int
    trust_level: str = "untrusted_external"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================
# PINNED HTTPS CONNECTION
# ============================================================

class PinnedHTTPSConnection(HTTPSConnection):
    """
    HTTPS connection that connects to a prevalidated IP.

    The original hostname is retained for:
        - TLS SNI
        - Certificate hostname verification
        - HTTP Host header

    The connection socket is established using the
    approved IP address rather than performing another
    DNS lookup for the destination hostname.
    """

    def __init__(
        self,
        host: str,
        *,
        pinned_ip: str,
        **kwargs: Any,
    ) -> None:

        self._pinned_ip = pinned_ip

        super().__init__(
            host,
            **kwargs,
        )

    def _new_conn(self):
        """
        Override urllib3 connection creation so the
        validated destination IP is used.
        """

        from urllib3.util.connection import (
            create_connection,
        )

        try:
            socket_connection = create_connection(
                (
                    self._pinned_ip,
                    self.port,
                ),
                self.timeout,
                source_address=self.source_address,
                socket_options=self.socket_options,
            )

            return socket_connection

        except OSError as exc:
            from urllib3.exceptions import (
                NewConnectionError,
            )

            raise NewConnectionError(
                self,
                "Unable to connect to the validated "
                "public IP address.",
            ) from exc


class PinnedHTTPSConnectionPool(HTTPSConnectionPool):
    """
    HTTPS connection pool using the pinned connection.
    """

    ConnectionCls = PinnedHTTPSConnection

    def __init__(
        self,
        host: str,
        *,
        pinned_ip: str,
        **kwargs: Any,
    ) -> None:

        self._pinned_ip = pinned_ip

        super().__init__(
            host,
            **kwargs,
        )

    def _new_conn(self):
        """
        Create a connection using the approved IP.
        """

        self.num_connections += 1

        connection = self.ConnectionCls(
            host=self.host,
            port=self.port,
            timeout=self.timeout,
            pinned_ip=self._pinned_ip,
            **self.conn_kw,
        )

        return connection


class PinnedHTTPSAdapter(HTTPAdapter):
    """
    Requests transport adapter using a pinned HTTPS IP.

    A separate session and adapter are created for
    each retrieval request to avoid accidentally
    reusing a connection for a different destination.
    """

    def __init__(
        self,
        pinned_ip: str,
        **kwargs: Any,
    ) -> None:

        self.pinned_ip = pinned_ip

        super().__init__(
            max_retries=0,
            **kwargs,
        )

    def init_poolmanager(
        self,
        connections: int,
        maxsize: int,
        block: bool = False,
        **pool_kwargs: Any,
    ) -> None:

        self.poolmanager = PoolManager(
            num_pools=connections,
            maxsize=maxsize,
            block=block,
            **pool_kwargs,
        )

        self.poolmanager.pool_classes_by_scheme[
            "https"
        ] = PinnedHTTPSConnectionPool

        self.poolmanager.connection_pool_kw[
            "pinned_ip"
        ] = self.pinned_ip


# ============================================================
# WEB RETRIEVER
# ============================================================

class WebRetriever:
    """
    Restricted HTTPS web retriever for AeroMind AI.

    Example:
        retriever = WebRetriever(max_chars=8000)

        document = retriever.retrieve(
            "https://docs.px4.io/"
        )

        print(document["title"])
        print(document["text"])
    """

    BLOCKED_HOST_SUFFIXES = (
        ".localhost",
        ".local",
        ".internal",
        ".test",
        ".invalid",
        ".example",
        ".home",
        ".lan",
    )

    BLOCKED_HOSTS = {
        "localhost",
        "metadata.google.internal",
    }

    REMOVABLE_TAGS = (
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "noscript",
        "iframe",
        "form",
        "button",
        "svg",
        "canvas",
        "aside",
        "template",
    )

    def __init__(
        self,
        max_chars: int = 8000,
        max_bytes: int = 1_000_000,
        connect_timeout: float = 5.0,
        read_timeout: float = 15.0,
    ) -> None:

        self.config = RetrieverConfig(
            max_chars=max_chars,
            max_bytes=max_bytes,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
        )

        self.max_chars = self.config.max_chars

        logger.info(
            "AeroMind WebRetriever initialised."
        )

    # --------------------------------------------------------
    # URL VALIDATION
    # --------------------------------------------------------

    def _validate_url(self, url: str) -> str:
        """
        Validate URL syntax and destination restrictions.

        Only HTTPS on port 443 is accepted.
        """

        if not isinstance(url, str):
            raise URLValidationError(
                "URL must be a string."
            )

        url = url.strip()

        if not url:
            raise URLValidationError(
                "URL cannot be empty."
            )

        if len(url) > self.config.max_url_length:
            raise URLValidationError(
                "URL exceeds the maximum permitted length."
            )

        if any(
            ord(character) < 33
            for character in url
        ):
            raise URLValidationError(
                "URL contains invalid characters."
            )

        try:
            parsed = urlsplit(url)

            port = parsed.port

        except ValueError as exc:
            raise URLValidationError(
                "Malformed URL."
            ) from exc

        if parsed.scheme.lower() != "https":
            raise URLValidationError(
                "Only HTTPS URLs are allowed."
            )

        if not parsed.hostname:
            raise URLValidationError(
                "URL must contain a hostname."
            )

        if parsed.username or parsed.password:
            raise URLValidationError(
                "Embedded URL credentials are prohibited."
            )

        if port not in (None, 443):
            raise URLValidationError(
                "Only HTTPS port 443 is allowed."
            )

        host = parsed.hostname.lower().rstrip(".")

        if host in self.BLOCKED_HOSTS:
            raise URLValidationError(
                "Local or internal hostname blocked."
            )

        if any(
            host.endswith(suffix)
            for suffix in self.BLOCKED_HOST_SUFFIXES
        ):
            raise URLValidationError(
                "Internal hostname suffix blocked."
            )

        if "." not in host:
            raise URLValidationError(
                "Single-label hostnames are prohibited."
            )

        # IP literals are not accepted. Public websites
        # must be accessed through DNS hostnames.
        try:
            ipaddress.ip_address(host)

        except ValueError:
            pass

        else:
            raise URLValidationError(
                "IP-literal URLs are not permitted."
            )

        if parsed.fragment:
            parsed = parsed._replace(fragment="")

        return urlunsplit(parsed)

    # --------------------------------------------------------
    # DNS RESOLUTION
    # --------------------------------------------------------

    def _resolve_public_addresses(
        self,
        hostname: str,
    ) -> list[str]:
        """
        Resolve hostname and accept only global addresses.

        If any resolved address is non-public, reject the
        entire destination.
        """

        try:
            entries = socket.getaddrinfo(
                hostname,
                443,
                family=socket.AF_UNSPEC,
                type=socket.SOCK_STREAM,
            )

        except socket.gaierror as exc:
            raise DNSResolutionError(
                "Unable to resolve the requested hostname."
            ) from exc

        if not entries:
            raise DNSResolutionError(
                "DNS returned no usable addresses."
            )

        addresses = []

        for entry in entries:

            raw_address = entry[4][0]

            try:
                address = ipaddress.ip_address(
                    raw_address
                )

            except ValueError as exc:
                raise DNSResolutionError(
                    "DNS returned an invalid IP address."
                ) from exc

            if not address.is_global:
                raise DNSResolutionError(
                    "Destination resolves to a "
                    "non-public network address."
                )

            normalised = str(address)

            if normalised not in addresses:
                addresses.append(normalised)

            if (
                len(addresses)
                > self.config.max_dns_addresses
            ):
                raise DNSResolutionError(
                    "DNS returned too many addresses."
                )

        return addresses

    # --------------------------------------------------------
    # HTML EXTRACTION
    # --------------------------------------------------------

    @staticmethod
    def _normalise_text(text: str) -> str:
        """Remove excessive whitespace."""

        return re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

    def _extract_html(
        self,
        html: bytes,
        url: str,
    ) -> RetrievedDocument:
        """
        Extract useful text and metadata from HTML.
        """

        soup = BeautifulSoup(
            html,
            "html.parser",
        )

        title = url

        if soup.title:
            extracted_title = soup.title.get_text(
                " ",
                strip=True,
            )

            if extracted_title:
                title = self._normalise_text(
                    extracted_title
                )[:300]

        for tag in soup(
            self.REMOVABLE_TAGS
        ):
            tag.decompose()

        main_content = (
            soup.find("main")
            or soup.find("article")
            or soup.body
            or soup
        )

        text = main_content.get_text(
            " ",
            strip=True,
        )

        text = self._normalise_text(text)

        if not text:
            raise RetrievalContentError(
                "The page contains no readable text."
            )

        text = text[:self.config.max_chars]

        domain = (
            urlsplit(url).hostname or ""
        ).lower()

        return RetrievedDocument(
            title=title,
            url=url,
            text=text,
            source_type="web_page",
            domain=domain,
            retrieved_at=datetime.now(
                timezone.utc
            ).isoformat(),
            content_type="text/html",
            character_count=len(text),
        )

    # --------------------------------------------------------
    # SECURE REQUEST
    # --------------------------------------------------------

    def _download_html(
        self,
        url: str,
        pinned_ip: str,
    ) -> bytes:
        """
        Download HTML from a prevalidated public IP.

        Redirects and proxies are disabled.
        """

        session = requests.Session()

        session.trust_env = False

        adapter = PinnedHTTPSAdapter(
            pinned_ip=pinned_ip,
        )

        session.mount(
            "https://",
            adapter,
        )

        response = None

        try:
            response = session.get(
                url,
                headers={
                    "User-Agent": (
                        self.config.user_agent
                    ),
                    "Accept": (
                        "text/html,"
                        "application/xhtml+xml"
                    ),
                    "Accept-Encoding": "identity",
                },
                timeout=(
                    self.config.connect_timeout,
                    self.config.read_timeout,
                ),
                allow_redirects=False,
                stream=True,
                verify=True,
            )

            if 300 <= response.status_code < 400:
                raise RetrievalHTTPError(
                    "HTTP redirects are not followed."
                )

            try:
                response.raise_for_status()

            except requests.HTTPError as exc:
                raise RetrievalHTTPError(
                    "Remote server returned HTTP "
                    f"{response.status_code}."
                ) from exc

            content_type = response.headers.get(
                "Content-Type",
                "",
            ).lower()

            if not (
                content_type.startswith("text/html")
                or content_type.startswith(
                    "application/xhtml+xml"
                )
            ):
                raise RetrievalContentError(
                    "Only HTML pages are supported."
                )

            declared_size = response.headers.get(
                "Content-Length"
            )

            if declared_size:
                try:
                    size = int(declared_size)

                except ValueError:
                    size = 0

                if size > self.config.max_bytes:
                    raise RetrievalSizeError(
                        "Web page exceeds maximum "
                        "download size."
                    )

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
                    > self.config.max_bytes
                ):
                    raise RetrievalSizeError(
                        "Downloaded content exceeds "
                        "the configured size limit."
                    )

                chunks.append(chunk)

            return b"".join(chunks)

        except requests.Timeout as exc:
            raise RetrievalTimeoutError(
                "The web retrieval request timed out."
            ) from exc

        except requests.SSLError as exc:
            raise RetrievalConnectionError(
                "HTTPS certificate verification failed."
            ) from exc

        except requests.ConnectionError as exc:
            raise RetrievalConnectionError(
                "Unable to establish a secure "
                "connection to the website."
            ) from exc

        except requests.RequestException as exc:
            raise RetrievalConnectionError(
                "An HTTP transport error occurred."
            ) from exc

        finally:
            if response is not None:
                response.close()

            session.close()

    # --------------------------------------------------------
    # PUBLIC RETRIEVAL API
    # --------------------------------------------------------

    def retrieve(
        self,
        url: str,
    ) -> dict[str, Any]:
        """
        Retrieve and extract a public HTTPS page.

        Returns:
            Dictionary containing:
                title
                url
                text
                source_type
                domain
                retrieved_at
                content_type
                character_count
                trust_level
        """

        validated_url = self._validate_url(
            url
        )

        hostname = urlsplit(
            validated_url
        ).hostname

        if hostname is None:
            raise URLValidationError(
                "URL hostname is missing."
            )

        addresses = self._resolve_public_addresses(
            hostname
        )

        logger.info(
            "Retrieving public HTTPS document: %s",
            hostname,
        )

        last_error = None

        for address in addresses:

            try:
                html = self._download_html(
                    validated_url,
                    pinned_ip=address,
                )

                document = self._extract_html(
                    html,
                    validated_url,
                )

                logger.info(
                    "Retrieved %d characters from %s.",
                    document.character_count,
                    document.domain,
                )

                return document.to_dict()

            except RetrievalConnectionError as exc:
                last_error = exc

                logger.warning(
                    "Connection to a validated "
                    "destination failed."
                )

            except RetrievalTimeoutError as exc:
                last_error = exc

                logger.warning(
                    "Validated destination timed out."
                )

        if last_error:
            raise last_error

        raise RetrievalError(
            "Unable to retrieve the requested page."
        )

    # --------------------------------------------------------
    # RAG INTEGRATION
    # --------------------------------------------------------

    def retrieve_for_rag(
        self,
        url: str,
    ) -> dict[str, Any]:
        """
        Retrieve a page in a RAG-ready structure.

        External text must be treated as untrusted data.
        """

        document = self.retrieve(url)

        return {
            "content": document["text"],
            "metadata": {
                "title": document["title"],
                "url": document["url"],
                "domain": document["domain"],
                "source_type": "web_page",
                "retrieved_at": (
                    document["retrieved_at"]
                ),
                "trust_level": (
                    "untrusted_external"
                ),
            },
        }


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

def main() -> int:
    """Run web retrieval from the command line."""

    parser = argparse.ArgumentParser(
        description=(
            "AeroMind AI Secure Web Retriever"
        )
    )

    parser.add_argument(
        "url",
        help="Public HTTPS URL to retrieve.",
    )

    parser.add_argument(
        "--max-chars",
        type=int,
        default=8000,
        help="Maximum extracted text characters.",
    )

    parser.add_argument(
        "--max-bytes",
        type=int,
        default=1_000_000,
        help="Maximum downloaded response bytes.",
    )

    parser.add_argument(
        "--rag",
        action="store_true",
        help="Return RAG-ready document format.",
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

    try:
        retriever = WebRetriever(
            max_chars=args.max_chars,
            max_bytes=args.max_bytes,
        )

        if args.rag:
            result = retriever.retrieve_for_rag(
                args.url
            )

        else:
            result = retriever.retrieve(
                args.url
            )

        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            )
        )

        return 0

    except RetrievalError as exc:

        logger.error(
            "Retrieval failed: %s",
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

    except ValueError as exc:

        print(
            json.dumps({
                "success": False,
                "error": str(exc),
                "error_type": "ConfigurationError",
            }),
            file=sys.stderr,
        )

        return 1

    except KeyboardInterrupt:

        logger.warning(
            "Retrieval interrupted by operator."
        )

        return 130


if __name__ == "__main__":
    sys.exit(main())
