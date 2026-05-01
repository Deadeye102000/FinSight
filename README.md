# FinSight

Production-grade AI stock research agent built incrementally with MCP, Claude, FastAPI, and Streamlit.

## Current MCP Tools

FinSight currently exposes these MCP tools from [server.py](/Users/Deadeye/Desktop/Projects/FinSight/finsight/mcp_server/server.py):

- `get_stock_price`
- `get_fundamentals`

Both tools accept U.S. and non-U.S. Yahoo Finance tickers such as `AAPL`, `MSFT`, `RELIANCE.NS`, and `TCS.NS`.

## Currency Semantics

FinSight is designed to be market-agnostic. It supports native exchange tickers while making currency handling explicit in the response payload.

### `get_stock_price`

Price outputs remain in the stock's native quote currency.

- `currency`: native Yahoo quote currency for the instrument
- `reporting_currency`: same as `currency` for price data
- `normalized_currency`: same as `currency` for price data because prices are not converted to USD

Example interpretation:

- `AAPL` -> `currency="USD"`
- `TCS.NS` -> `currency="INR"`

Fields such as `current_price`, `high_52w`, `low_52w`, and `ohlcv_last_5[*].open/high/low/close` should all be read in that native currency.

### `get_fundamentals`

Fundamentals preserve the source market currency metadata while normalizing large financial totals for easier cross-market comparison.

- `currency`: native Yahoo quote or reporting currency for the company
- `reporting_currency`: same as `currency`
- `normalized_currency`: always `USD`

Normalized fields:

- `market_cap` is returned in USD billions
- `revenue_ttm` is returned in USD billions

Non-normalized fields:

- ratios, margins, and percentages such as `pe_ratio`, `gross_margin`, `roe`, and `debt_to_equity` are unitless and are not FX-converted
- `target_price_mean` remains in the stock's native quote currency
- `eps_ttm` follows Yahoo's source reporting and should be interpreted in the company or listing currency context

If a stock is quoted in a non-USD currency, FinSight attempts Yahoo FX conversion to USD using direct pairs like `INRUSD=X` and then inverse-pair fallback like `USDINR=X`.

## Integration Notes

If you consume these tool responses downstream:

- use `currency` / `reporting_currency` when displaying native-market values
- use `normalized_currency` when displaying normalized aggregate fundamentals
- do not assume every numeric field is in USD
- for non-U.S. equities, prefer labeling both the native currency and the normalized currency in UI or reports
