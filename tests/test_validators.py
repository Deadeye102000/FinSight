"""Tests for shared validation helpers."""

import pytest

from finsight.mcp_server.utils.validators import (
    validate_peers_list,
    validate_symbol,
    validate_ticker,
)


def test_validate_ticker_rejects_non_string_and_long_symbols() -> None:
    assert validate_ticker(123) is False
    assert validate_ticker("A" * 21) is False


def test_validate_peers_list_rejects_bad_shapes() -> None:
    assert validate_peers_list("MSFT,GOOGL") is False
    assert validate_peers_list(["MSFT", 123]) is False
    assert validate_peers_list(["MSFT", ""]) is False
    assert validate_peers_list(["MSFT", "MSFT"]) is False


def test_validate_symbol_normalizes_and_rejects_empty() -> None:
    assert validate_symbol(" aapl ") == "AAPL"
    with pytest.raises(ValueError, match="Stock symbol is required"):
        validate_symbol("")
