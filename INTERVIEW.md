# FinSight — Interview Preparation

## How to use this file

Read each answer out loud before interviews. If you cannot explain it without reading, you do not know it yet. The goal is to answer every question in 90 seconds or less, naturally.

This file is intentionally grounded in the current codebase. If a feature is not implemented yet, the answer says that plainly instead of inventing a finished architecture.

---

## Q1: What is MCP and why did you use it instead of regular function calling?

**Current answer:**

MCP is a protocol for exposing tools through a separate server process so an agent can discover and call them through a standard interface. Regular Anthropic function calling means passing tool schemas directly to the Anthropic API and handling `tool_use` blocks inside the same application process. I chose MCP for FinSight because the financial data tools are already isolated behind a dedicated server: `finsight/mcp_server/server.py` creates a `FastMCP("FinSight")` server at line 25 and registers tools with `@server.tool()` at lines 28, 35, 42, 49, and 60. The tradeoff is extra process/transport complexity: right now the server runs over stdio at `finsight/mcp_server/server.py:67-70`, while `finsight/agent/orchestrator.py:1-8` is still only an Anthropic client placeholder and does not yet connect to MCP.

**Interview-safe phrasing today:**

"MCP is a standard way to expose tools through a separate server. Regular function calling would put the tool schemas directly into the Anthropic request and handle `tool_use` inside the app process. I chose MCP because FinSight has several independent financial data tools and the server already registers them cleanly with `@server.tool()` in `mcp_server/server.py`. The tradeoff is more moving parts, especially stdio transport; I accepted that because it keeps the research tools separate from the agent layer, though the orchestrator connection is still a later step."

**Update after Step 7:** The orchestrator now connects to MCP via stdio client in `finsight/agent/orchestrator.py:95-105`, initializes a session, lists tools, and executes them in a tool-calling loop with Claude at `orchestrator.py:115-180`. The tradeoff is accepted for clean separation, and the connection is fully implemented.

---

## Q2: Walk me through exactly what happens when a user types "Analyse TCS stock"

**Current answer: orchestrator implemented, UI/API integration pending.**

What exists today:

1. Streamlit does not yet accept a user query. It only sets page config and displays scaffold text in `finsight/ui/app.py:6-8`.
2. FastAPI does not yet expose a research endpoint. It only exposes `GET /health` in `finsight/api/main.py:9-12`.
3. `FinSightAgent.research()` now exists in `finsight/agent/orchestrator.py:25-85` and connects to MCP, runs Claude with tools, and returns a `ResearchReport`.
4. The system prompt is implemented in `orchestrator.py:15-35` and forces tool usage.
5. The MCP server tools are registered in `finsight/mcp_server/server.py:28-64`.

**What the flow becomes after Step 8:**

1. User input should hit a Streamlit text input or form in `finsight/ui/app.py`.
2. Streamlit should POST to a FastAPI research endpoint in `finsight/api/main.py`.
3. That endpoint should call `FinSightAgent.research()` in `finsight/agent/orchestrator.py`.
4. Claude receives the system prompt and calls tools based on the query.
5. For "Analyse TCS stock", Claude typically calls `get_stock_price`, `get_fundamentals`, `get_news_sentiment`, `get_corporate_announcements`, and possibly `compare_peers`: price gives technicals, fundamentals gives valuation, sentiment gives news tone, announcements gives India-specific corporate actions, and peers adds relative context.
6. The MCP server routes those calls through `finsight/mcp_server/server.py:28-64` to the implementations in `finsight/mcp_server/tools/`.
7. `yfinance` is called in `price.py:127-146` and `fundamentals.py:149-153`; BSE/NSE APIs are called in `announcements.py:125-149` and `announcements.py:308-332`.
8. Claude synthesizes a `ResearchReport` with structured markdown.
9. FastAPI should return JSON and Streamlit should render sections or tabs.

**The agent layer is now complete; UI/API wiring is the next step.**

---

## Q3: How do you prevent hallucinated financial data?

**Current answer: partial implementation, agent guardrails pending.**

A) **Grounding:** The data grounding exists at the tool layer, and now the agent layer enforces it. The tools fetch real provider data as before. The system prompt in `orchestrator.py:15-35` explicitly tells Claude "Be specific — cite actual numbers from the tools, never make up data." The `ResearchReport` schema in `orchestrator.py:9-21` separates tool outputs into dedicated fields like `price_data`, `fundamentals`, etc.

B) **Error propagation:** Every tool returns an `error` field, and the orchestrator now surfaces errors in the `ResearchReport.error` field at `orchestrator.py:75-76` if tool calls fail.

C) **Disclaimer:** The `ResearchReport` includes a forced `disclaimer` field set to "Not financial advice" in `orchestrator.py:77`, independent of Claude's output.

D) **Next eval step:** There is no `benchmarks/run_benchmarks.py` or `benchmarks/results.json` yet. The right next step is an eval harness that checks tool factuality and refusal behavior: for example, feed known tickers, invalid tickers, and missing-provider cases, then assert the report uses `error` fields instead of inventing P/E ratios, market caps, or announcement dates.

**Interview-safe phrasing today:**

"The grounding is strong at both layers: tools provide real data with error fields, and the agent now has a system prompt and schema that force Claude to only use tool outputs and always include a disclaimer. The orchestrator implementation in `orchestrator.py` makes this concrete."

---

## Q4: Why did you choose FinBERT over calling GPT-4 for sentiment?

**Current answer without benchmark numbers:**

I chose FinBERT for three reasons. First, cost: the model runs locally through HuggingFace, so the classification step costs zero per call after download; the code loads `ProsusAI/finbert` at `finsight/mcp_server/tools/sentiment.py:19-24` and lazily caches the pipeline at `sentiment.py:58-65`. Second, domain fit: the model is built for financial sentiment, and the tests verify the concrete behavior I need: a profit/growth sentence classifies positive and a bankruptcy/losses sentence classifies negative in `tests/test_sentiment.py:63-73`. Third, integration simplicity: `get_news_sentiment` maps FinBERT labels into signed scores at `sentiment.py:145-168` and returns a stable aggregate payload at `sentiment.py:171-209`.

**Numbers still pending:**

There is no `benchmarks/results.json` yet, so do not quote accuracy, latency per headline, or GPT-4 cost comparisons as measured project results. Once benchmarks exist, update this answer with measured FinBERT accuracy and latency.

**Tradeoff:**

FinBERT only returns positive, negative, or neutral. It does not explain why a headline is negative the way a generative model could. For FinSight's current use case, classification is enough because the downstream report needs aggregate tone, counts, and representative headlines rather than full narrative reasoning for every headline.

---

## Q5: How would you scale FinSight to 10,000 concurrent users?

**Current state:**

Right now FinSight does not have API rate limiting. `finsight/api/main.py:6-12` only defines the FastAPI app and a health check. Tool-level caching exists in specific places: yfinance history is cached with `@lru_cache(maxsize=64)` in `price.py:127-128`, fundamentals data is cached in `fundamentals.py:149-169`, FinBERT is cached as a module-level pipeline in `sentiment.py:23` and `sentiment.py:58-65`, and announcements use a 30-minute module-level cache in `announcements.py:68` and `announcements.py:543-547`. Peer comparison itself is concurrent rather than cached; it calls the underlying tools across tickers with `asyncio.gather` in `peers.py:112-114`.

**Path to scale:**

1. At 100 users, the first thing that breaks is provider I/O: yfinance and BSE/NewsAPI calls are external network dependencies with rate limits and variable latency.
2. At 1,000 users, I would add Redis caching keyed by ticker, period, filing or announcement type, and freshness window so repeated popular tickers like TCS, Reliance, Apple, and Microsoft do not hit providers repeatedly.
3. At 10,000 users, I would queue agent research jobs and stream progress back to the UI, because an LLM research request is a multi-tool workflow rather than a simple low-latency CRUD request.
4. The bottleneck is mostly I/O and third-party API limits, not CPU; FinBERT has compute cost, but it is cached in-process and can be batched or queued.
5. I would not try to horizontally scale by letting every API instance independently hammer yfinance, BSE, and NewsAPI; I would centralize cache and queueing first.

---

## Q6: How did you handle the BSE API rate limiting?

**Answer:**

BSE's public API does not publish a simple developer rate limit, so I was conservative. In `finsight/mcp_server/tools/announcements.py:29-31`, I set `HTTP_TIMEOUT_SECONDS = 5`, `BSE_REQUEST_DELAY_SECONDS = 1`, and `CACHE_TTL_SECONDS = 30 * 60`. The actual delay is implemented in `_sleep_before_bse_call()` at `announcements.py:115-122` using `time.monotonic()` and `time.sleep()`, and every BSE call goes through `_bse_get()` at `announcements.py:125-136`. I also added a module-level cache at `announcements.py:68` and read from it in `get_corporate_announcements()` at `announcements.py:543-547`, so repeated queries for the same `(ticker, announcement_type)` do not hit BSE for 30 minutes.

During testing, BSE returned a clean `200` JSON response for TCS announcements with rows under `Table`, including fields like `NEWSSUB`, `DT_TM`, `CATEGORYNAME`, `ATTACHMENTNAME`, and `SLONGNAME`. I also saw that NSE's fallback endpoint can return `200` with an empty `data` list, so the code treats "reachable" and "useful announcements found" as separate outcomes.

The behavior is covered by `tests/test_announcements.py`: BSE reachability is checked at lines 12-24, mock fallback on HTTP timeout is tested at lines 63-71, and cache speed is tested at lines 74-80.

---

## Q7: What is your context window strategy for long announcements?

**Current answer:**

BSE announcements can link to long PDFs, but the current implementation does not download or send PDF bodies to Claude. Instead, it keeps the Claude context compact by sending only structured announcement rows: date, category, and headline. That formatting happens in `_announcements_text()` at `finsight/mcp_server/tools/announcements.py:375-383`, and raw attachment URLs are preserved separately at `announcements.py:245-252` and `announcements.py:269-276`.

This means the current prompt size is bounded by the number of announcements `n`, not by PDF length. The tradeoff is that FinSight can summarize the announcement feed but cannot yet extract detailed financial figures from attached PDFs unless those figures appear in the headline. A better production approach would be to fetch the PDF attachment, extract text, chunk with overlap, and send only the financial highlights or results section to Claude.

**Important correction to the original template:**

There is no "first 2000 words" announcement strategy in the current code. That was true of the removed SEC MD&A implementation, not the current BSE/NSE announcements tool.

---

## Q8: What would you add if you had 2 more weeks?

**Answer:**

Three things. First, I would finish the agent/API/UI flow because right now the tools are strong, but a user cannot type "Analyse TCS stock" into Streamlit and get a complete research report; `ui/app.py:6-8`, `api/main.py:9-12`, and `agent/orchestrator.py:1-8` are still scaffolds. Second, I would build a shared Redis cache and provider adapter layer because the current caching is per-process and scattered across `@lru_cache` in price/fundamentals and a module dict in announcements, which does not work across multiple API instances. Third, I would add a proper eval pipeline: right now the tests verify tool behavior, like sentiment label sanity in `tests/test_sentiment.py:63-73` and announcements fallback/caching in `tests/test_announcements.py:63-80`, but a real eval would score complete generated reports for factuality, missing-data honesty, latency, and cost.

---

## Quick-fire answers (30 seconds each)

- **"What does stdio transport mean in MCP?"** → The MCP server communicates over standard input and output instead of HTTP. In FinSight, `server.run(transport="stdio")` is at `finsight/mcp_server/server.py:67-70`, which is simple for local agent-server wiring but not ideal for browser clients or horizontally scaled services.

- **"What is Wilder's RSI and why 14 periods?"** → RSI measures average gains versus average losses to estimate overbought or oversold momentum. FinSight computes Wilder-style smoothing manually in `price.py:66-90` and uses 14 periods in the result at `price.py:248` because 14 is the common default traders expect.

- **"What is a golden cross and why does it matter?"** → A golden cross is when a shorter moving average is above a longer moving average, often read as bullish trend confirmation. FinSight computes 50-day and 200-day moving averages at `price.py:225-226` and sets `golden_cross` with `ma_50 > ma_200` at `price.py:250-252`.

- **"Why asyncio.gather and not threading for peer comparison?"** → Peer comparison fetches several tickers at once, and each fetch is mostly provider I/O through the existing price and fundamentals tools. In `finsight/mcp_server/tools/peers.py:107-114`, I wrap the synchronous per-ticker fetch in `asyncio.to_thread()` and run those tasks with `asyncio.gather`, so the implementation reuses existing sync tools while avoiding sequential network waits.

- **"What is the difference between pe_ratio and peg_ratio?"** → P/E compares price to earnings, while PEG adjusts P/E by expected growth. FinSight returns both from Yahoo fields: `trailingPE` maps to `pe_ratio` and `pegRatio` maps to `peg_ratio` in `fundamentals.py:248-260`.

- **"Why FastAPI over Flask?"** → FastAPI is already the project scaffold at `api/main.py:3-12`; it gives typed request/response models and async-friendly endpoints, which fit a tool-heavy agent backend better than a minimal Flask app.

- **"Why Streamlit over a React frontend?"** → Streamlit is currently the fastest way to build a working research UI in Python; the scaffold is at `ui/app.py:3-8`. For an interview/demo project, speed of iteration matters more than custom frontend architecture.

- **"What does FinBERT stand for?"** → It is a financial-domain BERT model. In this project, the exact model is `ProsusAI/finbert`, configured at `sentiment.py:19-20` and loaded through HuggingFace at `sentiment.py:58-65`.

- **"What is the NewsAPI free tier limit?"** → Do not quote a limit from memory in interviews. The code only assumes a `NEWS_API_KEY` may or may not exist: missing keys return mock data with an explanatory error at `sentiment.py:222-230`.

- **"How does yfinance get data — does it scrape?"** → FinSight treats yfinance as an unofficial Yahoo Finance client. The code calls `yf.Ticker(...).history(...)` for prices at `price.py:127-133` and `yf.Ticker(ticker).info` for fundamentals at `fundamentals.py:149-153`, so provider reliability and schema changes are real risks.

---

## Red flags to avoid

These answers will end your interview:

- "I used it because it's popular" — always say why it fits YOUR use case.
- "I'm not sure how that part works" — if it is in the code, trace it before the interview.
- Giving numbers you can't back up — benchmark numbers must come from `benchmarks/results.json`, which does not exist yet.
- "The agent is intelligent" — say exactly what Claude does: reads tool outputs, follows the system prompt, and generates a structured report. Also say that this full orchestrator is not implemented yet.
- Overselling: "production-grade" means tests, error handling, observability, security posture, and deployment readiness. FinSight currently has tested tools and scaffolds, not a hedge-fund-ready production system.

---

## One-line project pitch (memorise this)

"FinSight is an AI research agent project for analysing Indian and US stocks together — fundamentals, technicals, news sentiment, Indian corporate announcements, and peer comparison — using Anthropic's MCP protocol with a local FinBERT model for zero-cost sentiment. I built it to understand how to design multi-tool LLM agents for real financial data, and the current codebase has the tools working while the full agent/UI orchestration is still being built."

The last sentence matters: it shows intellectual honesty and learning orientation, which senior interviewers respect more than overselling.

---

## Update Instructions For Future Steps

After completing each build step, come back to this file and update the relevant answers with real code references and line numbers.

Known pending updates:

- After Step 7, update Q1 and Q3 with the real MCP client connection and system prompt in `finsight/agent/orchestrator.py`.
- After Step 8, update Q2 with the real Streamlit -> FastAPI -> agent -> MCP -> report flow.
- After benchmarks are added, update Q4 with measured FinBERT accuracy, latency, and cost comparisons from `benchmarks/results.json`.
- After report schema work, update Q3 with the real `ResearchReport` disclaimer field and line number.
