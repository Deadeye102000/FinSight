"""Input validation helpers."""


def validate_symbol(symbol: str) -> str:
    """Validate and normalize a stock ticker symbol."""
    normalized = symbol.strip().upper()
    if not normalized:
        raise ValueError("Stock symbol is required.")
    return normalized
