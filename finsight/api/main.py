"""FastAPI application entrypoint for FinSight."""

import json
import logging
import time
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from starlette.middleware.base import BaseHTTPMiddleware

from finsight.agent.orchestrator import FinSightAgent, ResearchReport
from finsight.mcp_server.tools.announcements import get_corporate_announcements
from finsight.mcp_server.tools.fundamentals import get_fundamentals
from finsight.mcp_server.tools.peers import compare_peers
from finsight.mcp_server.tools.price import get_stock_price
from finsight.mcp_server.tools.sentiment import get_news_sentiment


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResearchRequest(BaseModel):
    query: str
    model_config = ConfigDict(str_max_length=500)


class ResearchResponse(BaseModel):
    success: bool
    data: Optional[ResearchReport] = None
    error: Optional[str] = None
    request_id: str
    timestamp: str


class ToolRequest(BaseModel):
    ticker: str
    peers: Optional[List[str]] = None
    company_name: Optional[str] = None
    include_sentiment: Optional[bool] = False


# Rate limiting storage: ip -> list of timestamps
rate_limit_store: Dict[str, List[float]] = defaultdict(list)


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        latency = time.time() - start_time
        logger.info(
            f"{request.method} {request.url.path} {response.status_code} {latency:.2f}s"
        )
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        timestamps = rate_limit_store[client_ip]
        # Remove timestamps older than 60 seconds
        timestamps[:] = [t for t in timestamps if now - t < 60]
        if len(timestamps) >= 10:
            return _format_error_response(
                request,
                "Rate limit exceeded: max 10 requests per minute",
                status.HTTP_429_TOO_MANY_REQUESTS,
            )
        timestamps.append(now)
        return await call_next(request)


app = FastAPI(title="FinSight API")

# Add middleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(RequestIDMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(RateLimitMiddleware)

# Initialize agent
agent = FinSightAgent("/Users/Deadeye/Desktop/Projects/FinSight/finsight/mcp_server/server.py")


def _format_error_response(request: Request, message: str, status_code: int) -> JSONResponse:
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    payload = {
        "success": False,
        "error": message,
        "request_id": request_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return JSONResponse(status_code=status_code, content=payload)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return _format_error_response(request, "Validation error", status.HTTP_422_UNPROCESSABLE_ENTITY)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    message = exc.detail if isinstance(exc.detail, str) else "Request error"
    return _format_error_response(request, message, exc.status_code)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return _format_error_response(request, "Internal server error", status.HTTP_500_INTERNAL_SERVER_ERROR)


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "ok"}


@app.post("/research", response_model=ResearchResponse)
async def research_endpoint(request: ResearchRequest, req: Request) -> ResearchResponse:
    """Main research endpoint that calls the agent."""
    query = request.query.strip()
    if len(query) < 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query must be at least 5 characters long.",
        )
    try:
        report = await agent.research(query)
        return ResearchResponse(
            success=True,
            data=report,
            error=None,
            request_id=req.state.request_id,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
    except Exception as e:
        logger.error(f"Research error: {e}", exc_info=True)
        return ResearchResponse(
            success=False,
            data=None,
            error=str(e),
            request_id=req.state.request_id,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )


@app.post("/tools/price")
async def price_tool_endpoint(request: ToolRequest, req: Request) -> Dict[str, Any]:
    """Call price tool directly."""
    if not request.ticker.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Ticker must be provided.",
        )
    try:
        result = get_stock_price(request.ticker)
        return {
            "success": True,
            "data": result,
            "request_id": req.state.request_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    except Exception as e:
        logger.error(f"Price tool error: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "request_id": req.state.request_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }


@app.post("/tools/fundamentals")
async def fundamentals_tool_endpoint(request: ToolRequest, req: Request) -> Dict[str, Any]:
    """Call fundamentals tool directly."""
    if not request.ticker.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Ticker must be provided.",
        )
    try:
        result = get_fundamentals(request.ticker)
        return {
            "success": True,
            "data": result,
            "request_id": req.state.request_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    except Exception as e:
        logger.error(f"Fundamentals tool error: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "request_id": req.state.request_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }


@app.post("/tools/sentiment")
async def sentiment_tool_endpoint(request: ToolRequest, req: Request) -> Dict[str, Any]:
    """Call sentiment tool directly."""
    if not request.ticker.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Ticker must be provided.",
        )
    try:
        company_name = request.company_name or request.ticker
        result = get_news_sentiment(request.ticker, company_name)
        return {
            "success": True,
            "data": result,
            "request_id": req.state.request_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    except Exception as e:
        logger.error(f"Sentiment tool error: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "request_id": req.state.request_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }


@app.post("/tools/filings")
async def filings_tool_endpoint(request: ToolRequest, req: Request) -> Dict[str, Any]:
    """Call filings tool directly."""
    if not request.ticker.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Ticker must be provided.",
        )
    try:
        result = get_corporate_announcements(request.ticker)
        return {
            "success": True,
            "data": result,
            "request_id": req.state.request_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    except Exception as e:
        logger.error(f"Filings tool error: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "request_id": req.state.request_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }


@app.post("/tools/peers")
async def peers_tool_endpoint(request: ToolRequest, req: Request) -> Dict[str, Any]:
    """Call peers tool directly."""
    if not request.ticker.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Ticker must be provided.",
        )
    try:
        peers = request.peers or ["MSFT", "GOOGL", "AMZN"]
        result = compare_peers(request.ticker, peers)
        return {
            "success": True,
            "data": result,
            "request_id": req.state.request_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    except Exception as e:
        logger.error(f"Peers tool error: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "request_id": req.state.request_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
