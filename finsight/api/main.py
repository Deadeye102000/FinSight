"""FastAPI application entrypoint for FinSight."""

from fastapi import FastAPI


app = FastAPI(title="FinSight API")


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "ok"}
