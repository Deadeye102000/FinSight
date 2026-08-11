"""Shared state schema for FinSight LangGraph orchestrator."""

from __future__ import annotations

from typing import Any, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class ResearchState(BaseModel):
    """Shared Pydantic v2 state schema for the FinSight research graph."""

    model_config = ConfigDict(arbitrary_types_allowed=True, str_strip_whitespace=True)

    ticker: str = Field(..., description="Target stock ticker symbol")
    peer_tickers: List[str] = Field(default_factory=list, description="Peer ticker symbols")

    # Fetch node payloads
    raw_price: Optional[dict[str, Any]] = None
    raw_fundamentals: Optional[dict[str, Any]] = None
    raw_news: Optional[dict[str, Any]] = None
    raw_peers: Optional[dict[str, Any]] = None

    # Parallel analysis sub-node outputs
    fundamentals_analysis: Optional[dict[str, Any]] = None
    sentiment_analysis: Optional[dict[str, Any]] = None
    peer_analysis: Optional[dict[str, Any]] = None

    # Thesis synthesis & final research report
    thesis: Optional[str] = None
    report: Optional[str] = None
    errors: List[str] = Field(default_factory=list)
