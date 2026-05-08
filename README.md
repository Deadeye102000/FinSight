# FinSight 📈 — AI Stock Research Agent with MCP

[![CI](https://github.com/Deadeye102000/FinSight/actions/workflows/ci.yml/badge.svg)](https://github.com/Deadeye102000/FinSight/actions/workflows/ci.yml)

> Autonomous AI agent that researches stocks using Model Context Protocol (MCP) with Claude or GPT.
> Reduces research time from 45 minutes to 90 seconds.

FinSight is an AI-powered stock research assistant for US and Indian equities. It combines price technicals, fundamentals, news sentiment, corporate announcements or filings, peer comparison, and an LLM-generated report into a Streamlit dashboard. The agent can run on Anthropic Claude or OpenAI GPT by changing one environment variable.

FinSight is a research assistant, not financial advice.

---

## Live Demo

- Deployed instance: `TODO: add Railway/HuggingFace Spaces URL after deployment`
- 2-minute demo GIF or YouTube link: `TODO: add demo link`

---

## Architecture

```text
User
  |
  v
Streamlit UI
  |  ticker, mode, peers
  v
FastAPI Backend
  |  /research + /tools/*
  v
FinSight Agent
  |  Claude or GPT tool-calling loop
  v
MCP Server
  |
  +--> get_stock_price ------------> yfinance
  +--> get_fundamentals -----------> yfinance
  +--> get_news_sentiment ---------> NewsAPI free tier + local FinBERT
  +--> get_corporate_announcements -> BSE/NSE announcements
  +--> compare_peers --------------> price + fundamentals tools
  |
  v
LLM synthesis
  |
  v
Structured research report + UI tabs
```

The current filing-style tool is implemented as BSE/NSE corporate announcements for Indian equities. A full SEC EDGAR 10-K/10-Q parser is listed in the roadmap.

---

## Features

- 5 MCP tools: price/technicals, fundamentals, news sentiment, filings or corporate announcements, peer comparison
- Dual LLM provider support: Anthropic Claude or OpenAI GPT
- Zero-cost sentiment analysis using local FinBERT model
- Free data sources: yfinance, BSE/NSE exchange announcements, NewsAPI free tier
- Supports US stocks (NYSE/NASDAQ) and Indian stocks (NSE/BSE)
- Running cost: approximately $1-3/month for light demo use, depending on provider/model usage
- Streamlit dashboard with Quick and Full research modes
- FastAPI backend with direct tool endpoints, request IDs, rate limiting, CORS, and safe JSON errors
- 92 passing tests with 81% coverage across API, agent, and MCP modules

---

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| UI | Streamlit | Fast demo-friendly Python UI for research workflows |
| API | FastAPI | Typed async-ready endpoints and clean OpenAPI ergonomics |
| Agent | Anthropic Claude or OpenAI GPT | Strong synthesis and tool-use reasoning with provider flexibility |
| Tool protocol | MCP | Standard boundary between agent and financial tools |
| Price data | yfinance | Free market data for US and international tickers |
| Fundamentals | yfinance | Free company ratios, margins, targets, and metadata |
| Sentiment model | ProsusAI/finbert | Local financial-domain sentiment classification |
| News source | NewsAPI | Free-tier headline retrieval when configured |
| Indian filings/events | BSE/NSE announcements | Exchange-native disclosures for Indian equities |
| Tests | pytest, pytest-asyncio, pytest-cov | Unit, async, integration, and coverage checks |
| Deployment | Docker, Docker Compose | Reproducible API/UI startup |

---

## Quick Start

Copy-paste setup:

```bash
git clone https://github.com/Deadeye102000/FinSight.git
cd FinSight

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

cp .env.example .env
```

Add keys to `.env` if you want live LLM and NewsAPI behavior:

```bash
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
FINSIGHT_LLM_PROVIDER=anthropic
OPENAI_MODEL=gpt-5
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
NEWS_API_KEY=your_key_here
```

To use GPT instead of Claude:

```bash
FINSIGHT_LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_key_here
OPENAI_MODEL=gpt-5
```

Start the backend:

```bash
.venv/bin/uvicorn finsight.api.main:app --reload --host 127.0.0.1 --port 8000
```

Start the UI in a second terminal:

```bash
.venv/bin/streamlit run ui/app.py
```

Open:

```text
http://localhost:8501
```

Docker Compose:

```bash
docker compose up --build
```

Then open:

```text
http://localhost:8501
```

Final pre-interview check:

```bash
.venv/bin/pytest tests/ -v && echo "ALL TESTS PASS — YOU'RE READY"
```

Coverage check:

```bash
.venv/bin/pytest tests/ -v --tb=short \
  --cov=finsight.mcp_server \
  --cov=finsight.agent \
  --cov=finsight.api \
  --cov-report=term-missing
```

---

## MCP Tools Reference

### 1. `get_stock_price`

Returns recent market data and technical indicators.

Input schema:

```json
{
  "ticker": "AAPL",
  "period": "3mo",
  "interval": "1d"
}
```

Output schema:

```json
{
  "ticker": "AAPL",
  "current_price": 0.0,
  "currency": "USD",
  "price_change_pct_30d": 0.0,
  "high_52w": 0.0,
  "low_52w": 0.0,
  "avg_volume_10d": 0,
  "rsi_14": 0.0,
  "macd_signal": "bullish | bearish | neutral",
  "ma_50": 0.0,
  "ma_200": 0.0,
  "golden_cross": false,
  "ohlcv_last_5": [],
  "error": null
}
```

### 2. `get_fundamentals`

Returns valuation ratios, profitability, leverage, margins, analyst rating, and target price.

Input schema:

```json
{
  "ticker": "AAPL"
}
```

Output schema:

```json
{
  "ticker": "AAPL",
  "company_name": "Apple Inc.",
  "sector": "Technology",
  "industry": "Consumer Electronics",
  "currency": "USD",
  "market_cap": 0.0,
  "pe_ratio": 0.0,
  "pb_ratio": 0.0,
  "gross_margin": 0.0,
  "net_margin": 0.0,
  "roe": 0.0,
  "debt_to_equity": 0.0,
  "analyst_rating": "Buy",
  "target_price_mean": 0.0,
  "valuation_summary": "Undervalued | Fairly Valued | Overvalued | Insufficient data",
  "error": null
}
```

### 3. `get_news_sentiment`

Returns headline sentiment using NewsAPI headlines and local FinBERT inference. If `NEWS_API_KEY` is missing, deterministic mock headlines are returned with an explanatory `error`.

Input schema:

```json
{
  "ticker": "AAPL",
  "company_name": "Apple",
  "n": 10
}
```

Output schema:

```json
{
  "ticker": "AAPL",
  "headlines_analysed": 10,
  "overall_sentiment": "Positive | Negative | Neutral",
  "sentiment_score": 0.0,
  "confidence": 0.0,
  "positive_count": 0,
  "negative_count": 0,
  "neutral_count": 0,
  "top_positive_headline": null,
  "top_negative_headline": null,
  "headlines": [],
  "error": null
}
```

### 4. `get_corporate_announcements`

Returns BSE/NSE corporate announcements and an investor-facing summary. This is the current filing-analysis implementation for Indian equities; SEC EDGAR support is a roadmap item.

Input schema:

```json
{
  "ticker": "TCS.NS",
  "announcement_type": "all",
  "n": 10
}
```

Output schema:

```json
{
  "ticker": "TCS.NS",
  "bse_code": "532540",
  "company_name": "Tata Consultancy Services",
  "announcements_fetched": 10,
  "announcement_types_found": ["results", "dividend"],
  "latest_quarterly_result": {},
  "upcoming_events": [],
  "claude_summary": "",
  "key_highlights": [],
  "sentiment": "Positive | Neutral | Concerning",
  "raw_announcements": [],
  "source": "BSE | NSE | mock",
  "error": null
}
```

### 5. `compare_peers`

Compares a main ticker with 2 to 5 peers using valuation and quality metrics.

Input schema:

```json
{
  "ticker": "AAPL",
  "peers": ["MSFT", "GOOGL", "AMZN"],
  "include_sentiment": false
}
```

Output schema:

```json
{
  "main_ticker": "AAPL",
  "comparison_date": "2026-05-04",
  "metrics_compared": ["pe_ratio", "roe", "net_margin"],
  "rankings": {},
  "winner_overall": "AAPL",
  "summary_table": [],
  "relative_valuation": "",
  "error": null
}
```

---

## API Reference

FastAPI endpoints:

- `GET /health`
- `POST /research`
- `POST /tools/price`
- `POST /tools/fundamentals`
- `POST /tools/sentiment`
- `POST /tools/filings`
- `POST /tools/peers`

Example:

```bash
curl -X POST http://127.0.0.1:8000/research \
  -H "Content-Type: application/json" \
  -d '{"query":"Analyse AAPL stock with fundamentals, technicals, sentiment, filings, and peers"}'
```

---

## Testing

Run all tests:

```bash
.venv/bin/pytest tests/ -v --tb=short
```

Run coverage:

```bash
.venv/bin/pytest tests/ -v --tb=short \
  --cov=finsight.mcp_server \
  --cov=finsight.agent \
  --cov=finsight.api \
  --cov-report=term-missing
```

Current local result:

```text
92 passed
81% total coverage
```

---

## Docker

Build the image:

```bash
docker build -t finsight .
```

Run both FastAPI and Streamlit in one container using Supervisor:

```bash
docker run --env-file .env -p 8000:8000 -p 8501:8501 finsight
```

Run split API/UI services:

```bash
docker compose up --build
```

---

## Deployment Checklist

- Create a Railway project from this repository.
- Set environment variables from `.env.example`.
- Deploy the API service on port `8000`.
- Deploy the UI service on port `8501`.
- Set `FINSIGHT_API_URL` in the UI service to the API service URL.
- Add the public URL to the Live Demo section above.
- Record a 2-minute demo and add the link above.

---

## Interview Q&A

Read the interview prep guide here:

[INTERVIEW.md](INTERVIEW.md)

---

## Current Limitations

- This is not a regulated investment advisory product.
- Full LLM research requires either `ANTHROPIC_API_KEY` with `FINSIGHT_LLM_PROVIDER=anthropic` or `OPENAI_API_KEY` with `FINSIGHT_LLM_PROVIDER=openai`.
- News sentiment uses mock data unless `NEWS_API_KEY` is configured.
- Most caches are in-process; production scaling should use Redis or another shared cache.
- yfinance and public exchange APIs can be stale, rate-limited, or schema-unstable.
- The current filing-analysis path uses BSE/NSE corporate announcements; full SEC EDGAR 10-K/10-Q parsing is not implemented yet.

---

## Resume Bullets

- Architected a multi-tool MCP server exposing 5 financial data tools orchestrated by Claude or GPT as an autonomous research agent, reducing stock research time from approximately 45 minutes to 90 seconds.
- Implemented zero-cost news sentiment classification using FinBERT local inference, processing real-time headlines without per-call LLM cost.
- Built data pipelines integrating yfinance, BSE/NSE exchange announcements, and NewsAPI with concurrent peer comparison across valuation and profitability metrics.
- Packaged FastAPI backend and Streamlit frontend with Docker, Docker Compose, and GitHub Actions CI.
- Achieved 81% test coverage across 92 unit and integration tests using pytest, pytest-asyncio, and pytest-cov.
