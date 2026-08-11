"""FastAPI application entrypoint for FinSight."""

import json
import logging
import os
import re
import tempfile
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.base import BaseHTTPMiddleware

from finsight.agent.orchestrator import FinSightAgent, ResearchReport
from finsight.mcp_server.tools.announcements import get_corporate_announcements
from finsight.mcp_server.tools.fundamentals import get_fundamentals
from finsight.mcp_server.tools.peers import compare_peers
from finsight.mcp_server.tools.price import get_stock_price
from finsight.mcp_server.tools.sentiment import get_news_sentiment
from finsight.rag.pipeline import RAGPipeline


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


class RAGQueryRequest(BaseModel):
    doc_id: str
    question: str = Field(min_length=5, max_length=500)
    k: int = Field(default=5, ge=1, le=10)


class RAGSource(BaseModel):
    text: str
    page_start: int
    page_end: int
    similarity: float
    retrieval_method: str
    excerpt_preview: str


class RAGResponse(BaseModel):
    success: bool
    doc_id: str
    question: str
    answer: str
    citations: list[int]
    sources: list[RAGSource]
    model_used: str
    tokens_used: int
    chunks_used: int
    error: Optional[str] = None


# Rate limiting storage: ip -> list of timestamps
rate_limit_store: Dict[str, List[float]] = defaultdict(list)
RATE_LIMIT_REQUESTS = int(os.getenv("FINSIGHT_RATE_LIMIT_REQUESTS", "10"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("FINSIGHT_RATE_LIMIT_WINDOW_SECONDS", "60"))
MAX_RAG_UPLOAD_BYTES = 20 * 1024 * 1024
DOC_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_]+$")


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
        timestamps[:] = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW_SECONDS]
        if len(timestamps) >= RATE_LIMIT_REQUESTS:
            retry_after = max(1, int(RATE_LIMIT_WINDOW_SECONDS - (now - timestamps[0])))
            response = _format_error_response(
                request,
                f"Rate limit exceeded: max {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW_SECONDS} seconds",
                status.HTTP_429_TOO_MANY_REQUESTS,
            )
            response.headers["Retry-After"] = str(retry_after)
            return response
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
rag_pipeline = RAGPipeline()


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


def _validate_doc_id(doc_id: str) -> str:
    doc_id = doc_id.strip()
    if not doc_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="doc_id is required.")
    if len(doc_id) > 50:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="doc_id must be 50 characters or fewer.")
    if not DOC_ID_PATTERN.fullmatch(doc_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="doc_id must contain only letters, numbers, and underscores.",
        )
    return doc_id


def _rag_sources(raw_sources: list[dict]) -> list[dict]:
    sources = []
    for source in raw_sources:
        metadata = source.get("metadata") or {}
        text = str(source.get("text") or "")
        sources.append(
            {
                "text": text,
                "page_start": int(metadata.get("page_start") or 0),
                "page_end": int(metadata.get("page_end") or metadata.get("page_start") or 0),
                "similarity": float(source.get("similarity") or 0.0),
                "retrieval_method": str(source.get("retrieval_method") or "dense"),
                "excerpt_preview": text[:300],
            }
        )
    return sources


@app.post("/rag/ingest")
async def rag_ingest_endpoint(
    file: UploadFile = File(...),
    doc_id: str = Form(...),
) -> Dict[str, Any]:
    """Upload and ingest a PDF for document Q&A."""
    doc_id = _validate_doc_id(doc_id)
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file must be a PDF.")
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="content_type must be application/pdf.")

    content = await file.read()
    if len(content) > MAX_RAG_UPLOAD_BYTES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="PDF must be 20MB or smaller.")

    already_existed = rag_pipeline.vector_store.document_exists(doc_id)
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            temp_path = tmp.name

        result = rag_pipeline.ingest(temp_path, doc_id)
        chunks_created = int(result.get("chunks_indexed", 0))
        chunks = rag_pipeline._chunk_cache.get(doc_id, [])
        tokens_estimated = sum(int(chunk.get("token_estimate") or 0) for chunk in chunks)
        return {
            "success": True,
            "doc_id": doc_id,
            "pages_extracted": int(result.get("pages_extracted", 0)),
            "chunks_created": chunks_created,
            "tokens_estimated": tokens_estimated,
            "already_existed": already_existed,
            "message": "Already ingested." if already_existed else "Ready to query.",
        }
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)


@app.post("/rag/query", response_model=RAGResponse)
async def rag_query_endpoint(request: RAGQueryRequest) -> RAGResponse:
    """Ask a grounded question against an ingested PDF."""
    doc_id = _validate_doc_id(request.doc_id)
    if not rag_pipeline.vector_store.document_exists(doc_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found. Ingest it first.",
        )

    result = rag_pipeline.query(doc_id, request.question, k=request.k)
    return RAGResponse(
        success=True,
        doc_id=doc_id,
        question=request.question,
        answer=result.get("answer") or "",
        citations=result.get("citations") or [],
        sources=[RAGSource(**source) for source in _rag_sources(result.get("sources") or [])],
        model_used=str(result.get("model_used") or "unknown"),
        tokens_used=int(result.get("tokens_used") or 0),
        chunks_used=int(result.get("chunks_used") or result.get("chunks_retrieved") or 0),
        error=result.get("error"),
    )


@app.get("/rag/documents")
async def rag_documents_endpoint() -> Dict[str, list[str]]:
    """List ingested RAG document IDs."""
    return {"documents": rag_pipeline.vector_store.list_documents()}


@app.delete("/rag/documents/{doc_id}")
async def rag_delete_document_endpoint(doc_id: str) -> Dict[str, Any]:
    """Delete an ingested RAG document."""
    doc_id = _validate_doc_id(doc_id)
    if not rag_pipeline.vector_store.document_exists(doc_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    rag_pipeline.vector_store.delete_document(doc_id)
    rag_pipeline._chunk_cache.pop(doc_id, None)
    rag_pipeline.retriever._bm25_index.pop(doc_id, None)
    rag_pipeline.retriever._corpus.pop(doc_id, None)
    return {"success": True, "message": "Deleted."}
