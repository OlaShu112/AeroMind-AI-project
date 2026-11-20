
"""
AeroMind AI - Intelligent Query Routing Engine
================================================

Module:
    ai.query_router

Purpose:
    Analyse user questions and determine the most
    appropriate knowledge retrieval sources.

Supported Routes:
    local       - AeroMind knowledge base and flight logs
    web         - Current public information
    research    - Academic research and scientific literature
    encyclopedia- General factual reference information
    general     - General LLM knowledge and reasoning

Features:
    - Rule-based intent classification
    - Weighted keyword and phrase matching
    - Multi-source routing
    - Confidence scoring
    - Query normalisation
    - Flight-related context detection
    - Temporal query detection
    - Academic research detection
    - Encyclopedia query detection
    - Local project knowledge detection
    - Configurable routing thresholds
    - Explainable routing decisions
    - Backward-compatible route() method
    - Structured JSON responses
    - Command-line interface
    - Safe handling of untrusted questions

Security:
    Query routing is advisory and read-only.

    This module must never:
        - Arm or disarm a drone
        - Execute flight commands
        - Modify PX4 parameters
        - Authorise autonomous missions
        - Override safety validation

Dependencies:
    Python standard library only.

Python:
    3.12+
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger(
    "aeromind.query_router"
)

logger.addHandler(logging.NullHandler())


# ============================================================
# EXCEPTIONS
# ============================================================

class QueryRouterError(Exception):
    """Base exception for query routing errors."""


class QueryValidationError(QueryRouterError):
    """Raised when a query is invalid."""


class QueryRoutingError(QueryRouterError):
    """Raised when routing cannot be completed."""


# ============================================================
# ROUTING CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class RouterConfig:
    """Configuration for AeroMind AI query routing."""

    max_query_length: int = 2000

    confidence_threshold: float = 0.35

    multi_source_threshold: float = 0.50

    max_sources: int = 3

    enable_multi_source: bool = True

    enable_encyclopedia: bool = True

    default_route: str = "general"

    def __post_init__(self) -> None:

        if self.max_query_length < 1:
            raise ValueError(
                "max_query_length must be positive."
            )

        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError(
                "confidence_threshold must be "
                "between 0 and 1."
            )

        if not 0 <= self.multi_source_threshold <= 1:
            raise ValueError(
                "multi_source_threshold must be "
                "between 0 and 1."
            )

        if self.max_sources < 1:
            raise ValueError(
                "max_sources must be positive."
            )


# ============================================================
# ROUTING RESULT MODEL
# ============================================================

@dataclass(frozen=True)
class RoutingDecision:
    """Structured query routing decision."""

    question: str

    primary_route: str

    routes: list[str]

    confidence: float

    scores: dict[str, float]

    reasons: list[str]

    requires_retrieval: bool

    is_time_sensitive: bool

    is_flight_related: bool

    is_research_related: bool

    timestamp: str

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================
# QUERY ROUTER
# ============================================================

class QueryRouter:
    """
    Intelligent knowledge-source routing for AeroMind AI.

    Examples:
        router = QueryRouter()

        route = router.route(
            "What is the latest PX4 release?"
        )

        print(route)
        # web

        decision = router.route_detailed(
            "Find recent scientific research "
            "on drone obstacle avoidance"
        )

        print(decision["routes"])
        # ['research', 'web']
    """

    ROUTES = (
        "local",
        "web",
        "research",
        "encyclopedia",
        "general",
    )

    # --------------------------------------------------------
    # INTENT KEYWORDS
    # --------------------------------------------------------

    WEB_TERMS = {
        "latest": 4.0,
        "today": 4.0,
        "current": 3.5,
        "currently": 3.5,
        "recent": 3.0,
        "news": 4.0,
        "this week": 4.0,
        "this month": 4.0,
        "new release": 3.5,
        "release notes": 3.0,
        "update": 2.0,
        "updates": 2.5,
        "new version": 3.0,
        "announcement": 2.5,
        "announced": 2.5,
        "breaking": 3.5,
        "live": 2.5,
        "as of now": 4.0,
        "recently": 3.0,
        "2026": 1.0,
    }

    RESEARCH_TERMS = {
        "research": 4.0,
        "scientific": 4.0,
        "paper": 3.5,
        "papers": 3.5,
        "publication": 3.5,
        "publications": 3.5,
        "journal": 4.0,
        "study": 3.0,
        "studies": 3.0,
        "peer reviewed": 4.0,
        "academic": 3.5,
        "literature review": 4.0,
        "systematic review": 4.0,
        "experiment": 2.5,
        "experimental": 2.5,
        "methodology": 2.5,
        "citation": 3.0,
        "citations": 3.0,
        "doi": 3.0,
        "openalex": 4.0,
        "scholarly": 3.5,
        "evidence based": 3.0,
    }

    LOCAL_TERMS = {
        "aeromind": 5.0,
        "our mission": 5.0,
        "our flight": 5.0,
        "my flight": 5.0,
        "my drone": 4.0,
        "our drone": 4.0,
        "flight log": 5.0,
        "flight logs": 5.0,
        "preflight": 4.0,
        "pre flight": 4.0,
        "our project": 4.0,
        "our system": 4.0,
        "our code": 5.0,
        "our simulation": 4.0,
        "my simulation": 4.0,
        "our controller": 5.0,
        "drone controller": 3.0,
        "first flight": 4.0,
        "our database": 4.0,
        "local knowledge": 4.0,
        "knowledge base": 3.5,
        "our documentation": 4.0,
        "our configuration": 4.0,
        "our telemetry": 5.0,
        "my telemetry": 5.0,
    }

    ENCYCLOPEDIA_TERMS = {
        "what is": 2.0,
        "what are": 2.0,
        "who is": 2.5,
        "who was": 2.5,
        "define": 3.0,
        "definition": 3.0,
        "history of": 3.0,
        "overview of": 2.5,
        "explain": 1.5,
        "meaning of": 2.5,
        "encyclopedia": 4.0,
        "wikipedia": 5.0,
        "background of": 2.5,
        "introduction to": 2.5,
    }

    FLIGHT_TERMS = {
        "drone",
        "flight",
        "takeoff",
        "take off",
        "landing",
        "land",
        "px4",
        "mavsdk",
        "mavlink",
        "gazebo",
        "sitl",
        "autopilot",
        "telemetry",
        "navigation",
        "waypoint",
        "obstacle avoidance",
        "flight controller",
        "geofence",
        "failsafe",
        "battery",
        "gps",
        "imu",
        "altitude",
        "mission planning",
        "uav",
        "quadcopter",
    }

    # --------------------------------------------------------
    # INITIALISATION
    # --------------------------------------------------------

    def __init__(
        self,
        config: Optional[RouterConfig] = None,
    ) -> None:

        self.config = config or RouterConfig()

        logger.info(
            "AeroMind QueryRouter initialised."
        )

    # --------------------------------------------------------
    # QUERY VALIDATION
    # --------------------------------------------------------

    def _validate_question(
        self,
        question: str,
    ) -> str:

        if not isinstance(question, str):
            raise QueryValidationError(
                "Question must be a string."
            )

        question = re.sub(
            r"\s+",
            " ",
            question,
        ).strip()

        if not question:
            raise QueryValidationError(
                "Question cannot be empty."
            )

        if (
            len(question)
            > self.config.max_query_length
        ):
            raise QueryValidationError(
                "Question exceeds the maximum "
                "permitted length."
            )

        if any(
            ord(character) < 32
            for character in question
        ):
            raise QueryValidationError(
                "Question contains invalid "
                "control characters."
            )

        return question

    # --------------------------------------------------------
    # NORMALISATION
    # --------------------------------------------------------

    @staticmethod
    def _normalise(
        question: str,
    ) -> str:
        """
        Normalise text for intent detection.
        """

        question = question.lower()

        question = re.sub(
            r"[^\w\s-]",
            " ",
            question,
        )

        question = re.sub(
            r"\s+",
            " ",
            question,
        )

        return question.strip()

    # --------------------------------------------------------
    # KEYWORD MATCHING
    # --------------------------------------------------------

    @staticmethod
    def _match_terms(
        text: str,
        terms: dict[str, float],
    ) -> tuple[float, list[str]]:
        """
        Match complete words and phrases.

        Avoid substring matches such as:
            'paper' inside 'paperwork'
            'land' inside 'landscape'
        """

        score = 0.0
        matches = []

        for term, weight in terms.items():

            pattern = (
                r"(?<!\w)"
                + re.escape(term)
                + r"(?!\w)"
            )

            if re.search(pattern, text):

                score += weight

                matches.append(term)

        return score, matches

    @staticmethod
    def _match_set(
        text: str,
        terms: set[str],
    ) -> list[str]:

        matches = []

        for term in terms:

            pattern = (
                r"(?<!\w)"
                + re.escape(term)
                + r"(?!\w)"
            )

            if re.search(pattern, text):
                matches.append(term)

        return sorted(matches)

    # --------------------------------------------------------
    # SCORE CALCULATION
    # --------------------------------------------------------

    @staticmethod
    def _normalise_scores(
        raw_scores: dict[str, float],
    ) -> dict[str, float]:
        """
        Convert weighted scores to relative confidence
        values between 0 and 1.

        These are heuristic scores, not calibrated
        statistical probabilities.
        """

        maximum = max(
            raw_scores.values(),
            default=0.0,
        )

        if maximum <= 0:
            return {
                route: 0.0
                for route in raw_scores
            }

        return {
            route: round(
                min(score / maximum, 1.0),
                4,
            )
            for route, score in raw_scores.items()
        }

    # --------------------------------------------------------
    # ROUTING ANALYSIS
    # --------------------------------------------------------

    def _analyse(
        self,
        question: str,
    ) -> RoutingDecision:

        original_question = (
            self._validate_question(question)
        )

        text = self._normalise(
            original_question
        )

        web_score, web_matches = (
            self._match_terms(
                text,
                self.WEB_TERMS,
            )
        )

        research_score, research_matches = (
            self._match_terms(
                text,
                self.RESEARCH_TERMS,
            )
        )

        local_score, local_matches = (
            self._match_terms(
                text,
                self.LOCAL_TERMS,
            )
        )

        encyclopedia_score, encyclopedia_matches = (
            self._match_terms(
                text,
                self.ENCYCLOPEDIA_TERMS,
            )
        )

        flight_matches = self._match_set(
            text,
            self.FLIGHT_TERMS,
        )

        is_time_sensitive = bool(
            web_matches
        )

        is_research_related = bool(
            research_matches
        )

        is_flight_related = bool(
            flight_matches
        )

        # Flight terminology alone does not prove
        # that the question concerns local project data.
        # However, a small local score helps when
        # combined with AeroMind-specific terminology.

        if is_flight_related and local_matches:
            local_score += 1.0

        # Scientific questions involving recent
        # publications may benefit from both academic
        # and web retrieval.

        if (
            is_research_related
            and is_time_sensitive
        ):
            research_score += 1.0
            web_score += 1.0

        # Explicit local project references take
        # precedence over generic explanations.

        if any(
            term in local_matches
            for term in (
                "aeromind",
                "our mission",
                "our flight",
                "my flight",
                "flight log",
                "flight logs",
                "our code",
                "our telemetry",
                "my telemetry",
            )
        ):
            local_score += 2.0

        raw_scores = {
            "local": local_score,
            "web": web_score,
            "research": research_score,
            "encyclopedia": (
                encyclopedia_score
                if self.config.enable_encyclopedia
                else 0.0
            ),
        }

        scores = self._normalise_scores(
            raw_scores
        )

        ranked_routes = sorted(
            raw_scores,
            key=lambda route: (
                -raw_scores[route],
                (
                    "local",
                    "research",
                    "web",
                    "encyclopedia",
                ).index(route),
            ),
        )

        highest_route = ranked_routes[0]
        highest_score = raw_scores[
            highest_route
        ]

        reasons = []

        if web_matches:
            reasons.append(
                "Time-sensitive terms detected: "
                + ", ".join(web_matches)
            )

        if research_matches:
            reasons.append(
                "Academic research terms detected: "
                + ", ".join(research_matches)
            )

        if local_matches:
            reasons.append(
                "Local project terms detected: "
                + ", ".join(local_matches)
            )

        if encyclopedia_matches:
            reasons.append(
                "General reference terms detected: "
                + ", ".join(
                    encyclopedia_matches
                )
            )

        if flight_matches:
            reasons.append(
                "Drone-related terminology detected: "
                + ", ".join(flight_matches)
            )

        # No sufficiently strong retrieval intent.
        if highest_score <= 0:

            primary_route = (
                self.config.default_route
            )

            selected_routes = [
                primary_route
            ]

            confidence = 0.0

            reasons.append(
                "No specific retrieval source "
                "was identified."
            )

        else:

            primary_route = highest_route

            confidence = scores[
                primary_route
            ]

            selected_routes = [
                primary_route
            ]

            if self.config.enable_multi_source:

                for route in ranked_routes[1:]:

                    if (
                        len(selected_routes)
                        >= self.config.max_sources
                    ):
                        break

                    if raw_scores[route] <= 0:
                        continue

                    if (
                        scores[route]
                        >= self.config.multi_source_threshold
                    ):
                        selected_routes.append(
                            route
                        )

            reasons.append(
                f"Primary knowledge source: "
                f"{primary_route}."
            )

        requires_retrieval = any(
            route != "general"
            for route in selected_routes
        )

        return RoutingDecision(
            question=original_question,
            primary_route=primary_route,
            routes=selected_routes,
            confidence=confidence,
            scores=scores,
            reasons=reasons,
            requires_retrieval=requires_retrieval,
            is_time_sensitive=is_time_sensitive,
            is_flight_related=is_flight_related,
            is_research_related=is_research_related,
            timestamp=datetime.now(
                timezone.utc
            ).isoformat(),
            metadata={
                "router": "AeroMind QueryRouter",
                "version": "2.0",
                "routing_method": (
                    "weighted_rule_based"
                ),
                "confidence_type": (
                    "relative_heuristic"
                ),
                "flight_execution_allowed": False,
            },
        )

    # --------------------------------------------------------
    # BACKWARD-COMPATIBLE ROUTING
    # --------------------------------------------------------

    def route(
        self,
        question: str,
    ) -> str:
        """
        Return the primary route as a string.

        Maintains compatibility with the original
        QueryRouter implementation.
        """

        decision = self._analyse(
            question
        )

        return decision.primary_route

    # --------------------------------------------------------
    # DETAILED ROUTING
    # --------------------------------------------------------

    def route_detailed(
        self,
        question: str,
    ) -> dict[str, Any]:
        """
        Return the complete routing decision.
        """

        decision = self._analyse(
            question
        )

        return decision.to_dict()

    # --------------------------------------------------------
    # MULTI-SOURCE ROUTING
    # --------------------------------------------------------

    def route_sources(
        self,
        question: str,
    ) -> list[str]:
        """
        Return all selected knowledge sources.
        """

        decision = self._analyse(
            question
        )

        return decision.routes

    # --------------------------------------------------------
    # ROUTING EXPLANATION
    # --------------------------------------------------------

    def explain(
        self,
        question: str,
    ) -> str:
        """
        Explain why a particular route was selected.
        """

        decision = self._analyse(
            question
        )

        explanation = [
            "AeroMind AI Query Routing Decision",
            "-----------------------------------",
            f"Question: {decision.question}",
            f"Primary route: {decision.primary_route}",
            "Selected sources: "
            + ", ".join(decision.routes),
            f"Relative score: {decision.confidence:.2f}",
            "",
            "Reasons:",
        ]

        for reason in decision.reasons:
            explanation.append(
                f"- {reason}"
            )

        return "\n".join(
            explanation
        )


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

def main() -> int:
    """Run query routing from the command line."""

    parser = argparse.ArgumentParser(
        description=(
            "AeroMind AI Intelligent Query Router"
        )
    )

    parser.add_argument(
        "question",
        nargs="+",
        help="Question to analyse.",
    )

    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Return complete JSON routing metadata.",
    )

    parser.add_argument(
        "--sources",
        action="store_true",
        help="Return all selected knowledge sources.",
    )

    parser.add_argument(
        "--explain",
        action="store_true",
        help="Explain the routing decision.",
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

    question = " ".join(
        args.question
    )

    router = QueryRouter()

    try:

        if args.detailed:

            output = router.route_detailed(
                question
            )

            print(
                json.dumps(
                    output,
                    indent=2,
                    ensure_ascii=False,
                )
            )

        elif args.sources:

            output = router.route_sources(
                question
            )

            print(
                json.dumps(
                    output,
                    indent=2,
                    ensure_ascii=False,
                )
            )

        elif args.explain:

            print(
                router.explain(question)
            )

        else:

            print(
                router.route(question)
            )

        return 0

    except QueryRouterError as exc:

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
            "Query routing interrupted."
        )

        return 130


if __name__ == "__main__":
    sys.exit(main())
