"""Tests for the FinSight stock price tool and MCP server bootstrap."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from finsight.mcp_server.tools.price import get_stock_price
from finsight.mcp_server.utils.validators import validate_period, validate_ticker


ROOT = Path(__file__).resolve().parents[1]


def test_valid_us_ticker() -> None:
    result = get_stock_price("AAPL", "1mo")
    assert result["error"] is None
    assert result["current_price"] > 0


def test_valid_indian_ticker() -> None:
    result = get_stock_price("RELIANCE.NS", "1mo")
    assert result["error"] is None
    assert result["current_price"] > 0


def test_invalid_ticker_returns_error() -> None:
    result = get_stock_price("INVALID_TICKER_XYZ", "1mo")
    assert result["error"] is not None


def test_rsi_range() -> None:
    result = get_stock_price("AAPL", "1mo")
    assert result["error"] is None
    assert 0 <= result["rsi_14"] <= 100


def test_golden_cross_is_bool() -> None:
    result = get_stock_price("AAPL", "1mo")
    assert result["error"] is None
    assert isinstance(result["golden_cross"], bool)


def test_ohlcv_has_5_entries() -> None:
    result = get_stock_price("AAPL", "1mo")
    assert result["error"] is None
    assert len(result["ohlcv_last_5"]) == 5


def test_validator_rejects_empty() -> None:
    assert validate_ticker("") is False


def test_validator_rejects_spaces() -> None:
    assert validate_ticker("APPLE INC") is False


def test_invalid_period() -> None:
    assert validate_period("10y") is False


def test_mcp_server_starts() -> None:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(ROOT) if not pythonpath else f"{ROOT}{os.pathsep}{pythonpath}"

    process = subprocess.Popen(
        [sys.executable, "-m", "finsight.mcp_server.server"],
        cwd=ROOT,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        time.sleep(1.5)
        exit_code = process.poll()
        stderr_output = ""
        if exit_code is not None:
            stderr_output = process.stderr.read() if process.stderr else ""
        assert exit_code is None, f"Server exited early with code {exit_code}: {stderr_output}"
    finally:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)
