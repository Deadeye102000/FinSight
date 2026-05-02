# FinSight

FinSight is an incremental AI stock research agent built around MCP tools, with FastAPI and Streamlit scaffolds ready for later application layers.

## Current Status

The active implementation is the MCP server in `finsight/mcp_server/server.py`.

Registered MCP tools:

- `get_stock_price`
- `get_fundamentals`
- `get_news_sentiment`
- `get_corporate_announcements`
- `compare_peers`

Application scaffolds:

- FastAPI exposes `GET /health` from `finsight/api/main.py`
- Streamlit currently shows a basic project-ready page from `finsight/ui/app.py`

## Tool Summary

### `get_stock_price`

Fetches Yahoo Finance market data and technical indicators for U.S. and non-U.S. tickers.

Input:

- `ticker: str`
- `period: str = "3mo"` where supported values are `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`
- `interval: str = "1d"`

Output includes:

- current price
- native quote currency
- 30-day price change percentage
- 52-week high and low
- 10-day average volume
- RSI 14
- MACD signal
- 50-day and 200-day moving averages
- golden cross flag
- last 5 OHLCV rows
- `error`

### `get_fundamentals`

Fetches Yahoo Finance company fundamentals and valuation metrics.

Input:

- `ticker: str`

Output includes:

- company name
- sector and industry
- currency metadata
- market cap
- valuation ratios
- EPS and revenue
- margins
- ROE and ROA
- debt and liquidity ratios
- dividend yield
- analyst rating
- target price
- valuation summary
- `error`

### `get_news_sentiment`

Fetches recent English headlines from NewsAPI and scores them locally with `ProsusAI/finbert`. FinBERT is loaded lazily on the first call and cached in-process, so sentiment classification does not use a paid API.

Input:

- `ticker: str`
- `company_name: str`
- `n: int = 10` where valid values are `1` through `50`

Output includes:

- number of headlines analysed
- overall sentiment: `Positive`, `Negative`, or `Neutral`
- sentiment score from `-1.0` to `1.0`
- confidence from `0.0` to `1.0`
- positive, negative, and neutral counts
- top positive and negative headlines
- per-headline sentiment results
- `error`

If `NEWS_API_KEY` is not set, the tool returns deterministic mock headlines plus an explanatory error message instead of crashing.

### `get_corporate_announcements`

Fetches recent BSE corporate announcements for Indian equities and summarizes investor-relevant items. The tool supports common NSE tickers such as `TCS.NS`, resolves them to BSE codes, falls back to NSE announcements where useful, and uses deterministic mock data if exchange APIs are unreachable.

Input:

- `ticker: str`
- `announcement_type: str = "all"` where supported values are `results`, `dividend`, `merger`, and `all`
- `n: int = 10`

Output includes:

- BSE code
- company name
- fetched announcement count
- announcement types found
- latest quarterly result metadata where available
- upcoming board meeting, dividend, or record-date events
- Claude or fallback summary
- top highlights
- announcement sentiment: `Positive`, `Neutral`, or `Concerning`
- raw announcement rows with attachment URLs
- source: `BSE`, `NSE`, or `mock`
- `error`

Results are cached in-process by ticker and announcement type for 30 minutes. Claude Haiku is used when `ANTHROPIC_API_KEY` is configured; otherwise the tool returns a deterministic local summary.

### `compare_peers`

Compares a main ticker against 2 to 5 peer tickers by calling the existing price and fundamentals tools concurrently. Sentiment can also be included when needed, but it is optional because it is slower.

Input:

- `ticker: str`
- `peers: list[str]` with 2 to 5 unique peer tickers
- `include_sentiment: bool = False`

Output includes:

- comparison date
- metrics compared
- per-metric rankings
- overall winner
- summary table with valuation, profitability, leverage, RSI, and optional sentiment fields
- relative valuation summary
- `error`

Ranking rules:

- lower is better for `pe_ratio`, `pb_ratio`, and `debt_to_equity`
- higher is better for `roe`, `gross_margin`, `net_margin`, and RSI below 70
- composite score is based on per-metric ranks; the lower total is better

## Currency Semantics

FinSight is designed to be market-agnostic. It supports native exchange tickers while making currency handling explicit in response payloads.

### Price Data

`get_stock_price` returns prices in the stock's native quote currency.

- `currency`: native Yahoo quote currency for the instrument
- `reporting_currency`: same as `currency`
- `normalized_currency`: same as `currency` because prices are not converted to USD

Examples:

- `AAPL` -> `currency="USD"`
- `TCS.NS` -> `currency="INR"`

Fields such as `current_price`, `high_52w`, `low_52w`, and `ohlcv_last_5[*].open/high/low/close` should be read in that native currency.

### Fundamentals Data

`get_fundamentals` preserves source market currency metadata while normalizing large financial totals for easier cross-market comparison.

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

## Configuration

Create a local `.env` from `.env.example` when running the full project.

Relevant environment variables:

- `NEWS_API_KEY`: optional for tests and local mock sentiment, required for live NewsAPI headlines
- `ANTHROPIC_API_KEY`: optional for local fallback, required for Claude-generated announcement summaries
- `MCP_SERVER_HOST`
- `MCP_SERVER_PORT`
- `API_HOST`
- `API_PORT`

NewsAPI offers a free tier at `newsapi.org`. The sentiment model itself is local through HuggingFace Transformers and PyTorch.

## Local Setup

Install dependencies into a virtual environment:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Run the MCP server over stdio:

```bash
.venv/bin/python -m finsight.mcp_server.server
```

Run the FastAPI scaffold:

```bash
.venv/bin/uvicorn finsight.api.main:app --reload
```

Run the Streamlit scaffold:

```bash
.venv/bin/streamlit run finsight/ui/app.py
```

## Tests

Run all tests:

```bash
.venv/bin/pytest
```

Run the sentiment test suite:

```bash
.venv/bin/pytest tests/test_sentiment.py -v --tb=short
```

Run the corporate announcements test suite:

```bash
.venv/bin/pytest tests/test_announcements.py -v --tb=short
```

Run the peer comparison test suite:

```bash
.venv/bin/pytest tests/test_peers.py -v --tb=short
```

The first sentiment run may take longer because `ProsusAI/finbert` is downloaded and cached locally.

## Integration Notes

If you consume these tool responses downstream:

- use `currency` and `reporting_currency` when displaying native-market values
- use `normalized_currency` when displaying normalized aggregate fundamentals
- do not assume every numeric field is in USD
- handle `error` on every tool response
- for non-U.S. equities, prefer labeling both the native currency and the normalized currency in UI or reports
- treat NewsAPI availability separately from sentiment availability because FinBERT runs locally once downloaded
- treat BSE/NSE availability and Claude availability separately because exchange retrieval and announcement summarization can fail independently
