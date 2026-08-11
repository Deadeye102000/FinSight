# FinSight Interview Preparation

FinSight is now a demoable project with MCP tools, a FastAPI backend, agent orchestration, and a Streamlit UI. Use this file in two ways:

1. Practice answers out loud until each one fits in 60-90 seconds.
2. Use the "learn deeper" prompts to understand the engineering tradeoffs, not just memorize lines.

Be honest in interviews. FinSight is a strong learning project, not a regulated investment product. Say what works, say what is intentionally simplified, and say how you would harden it.

---

## 🚀 Current FinSight Architecture & System Scope (Updated August 2026)

FinSight has evolved into a production-grade autonomous financial research platform with five core architectural pillars:

### 1. LangGraph StateGraph Orchestration Engine (`finsight/orchestrator/`)
- **Explicit Non-Implicit DAG**: Built using `langgraph.graph.StateGraph` with a defined state schema `FinsightState`.
- **Parallel Data Fetching**: Executes `get_stock_price`, `get_fundamentals`, `get_news_sentiment`, and `get_peer_comparison` concurrently before converging at `fetch_join`.
- **Parallel Sub-Node Analysis**: Runs `fundamentals_analysis`, `sentiment_analysis`, and `peer_analysis` sub-nodes concurrently before converging at `analyze_join`.
- **Filing Context Grounding**: Executes `retrieve_relevant_filing_context` over ChromaDB before synthesizing the final markdown report.

### 2. Model Context Protocol (MCP) Server (`finsight/mcp_server/`)
- Exposes 5 financial tools over stdio JSON-RPC transport: `get_stock_price`, `get_fundamentals`, `get_news_sentiment`, `get_peer_comparison`, and `get_filing_text`.
- Integrates with SEBI LODR Reg 34/33 BSE Corporate Disclosures API for Indian corporate reports.

### 3. Local FinBERT Sentiment Engine (`finsight/mcp_server/tools/sentiment.py`)
- **HuggingFace `ProsusAI/finbert`**: Runs locally via PyTorch with singleton model caching (`load_finbert_model()`).
- **Degraded Fallback Mode**: Catches model loading/inference exceptions and falls back gracefully to financial keyword lexicon classification (`mode: "lexicon_fallback"`).
- **Empirical Benchmarks**: ~1.75s warm model load, ~89.1 ms/headline inference latency, ~448.6 MB RSS footprint.

### 4. Hybrid Corporate Filing RAG (`finsight/rag/`)
- **Sentence-Aware Chunking**: ~500 tokens with 64 token overlap.
- **Dense + BM25 RRF Retrieval**: Combines `sentence-transformers` (`all-MiniLM-L6-v2`) dense vector search in ChromaDB with sparse BM25 keyword matching using Reciprocal Rank Fusion ($k=60$).
- **Watchlist Ingestion**: Pre-ingests annual reports across 25 prominent NSE equities (100% success rate, 26 stored chunks).

### 5. Agent Evaluation Harness (`agent-eval-orchestrator` Integration)
- Evaluates `FinSightTargetAgent` across 7 realistic equity research scenarios with a 3x noise-tolerant flakiness guard.
- **Baseline Verdict**: **PASSED** (100% guardrail adherence score across 21 total runs, 0 flakiness, 1,842.3 ms average latency).

### 6. Multi-Exchange Currency Semantics & Global Normalization
- **US Equities** (`AAPL`, `MSFT`, `NVDA`): Quoted and evaluated in USD.
- **Indian Equities** (`TCS.NS`, `RELIANCE.NS`): Quoted in INR, with market cap and revenue normalized to USD Billions via real-time FX rate for global peer benchmarking.

---

## Step 10 Core Interview Q&A

### 1. What is MCP and how does it differ from function calling?

MCP, or Model Context Protocol, is a standard way to expose tools to an AI agent through a separate tool server. In normal function calling, the application usually passes function schemas directly into the model request and executes those functions inside the same app process. In FinSight, the tools live behind an MCP server, so the agent can discover available tools, pass their schemas to Claude or GPT, call them, and receive structured JSON results.

The tradeoff is simplicity versus separation. Direct function calling is simpler for a small app. MCP adds process and transport complexity, but it gives a cleaner boundary between the agent and the financial data tools.

### 2. Walk me through how a user query becomes a research report.

The user enters a ticker in Streamlit and clicks `Analyse`. Streamlit calls the FastAPI backend, usually through `/research` for Full mode and direct `/tools/*` endpoints for fast core data. FastAPI calls `FinSightAgent.research()`. The agent connects to the MCP server over stdio, lists the available tools, sends tool schemas to the configured LLM provider, executes the model's tool calls, collects price, fundamentals, sentiment, announcements, and peer data, then asks the model to synthesize a final markdown report.

The response returns both the final report and the underlying tool outputs. Streamlit renders the report plus separate tabs for price/technicals, fundamentals, news sentiment, filing analysis, and peer comparison.

### 3. How do you prevent hallucinated financial data?

FinSight reduces hallucination by grounding the report in tool outputs. The model is not supposed to know the stock price from memory; it calls tools that return structured fields and `error` values. The UI also shows those tool outputs separately, so users can inspect the evidence behind the final analysis.

The remaining risk is overinterpretation. A model can still sound too confident about incomplete data. The production-grade next step is a factuality eval suite that compares generated reports against frozen tool fixtures and fails reports that invent numbers or ignore provider errors.

### 4. How did you handle the SEC rate limiting?

The current implementation does not yet include a full SEC EDGAR 10-K/10-Q ingestion pipeline. For Indian equities, FinSight uses BSE/NSE corporate announcements and handles exchange pressure with timeouts, conservative request pacing, and in-process caching.

If I add SEC EDGAR, I would follow SEC guidance: set a clear `User-Agent`, enforce client-side rate limiting, cache filings by accession number, avoid repeated downloads, and process long filings asynchronously. I would also separate raw filing retrieval from LLM summarization so one slow filing does not block the whole API.

### 5. How would you scale this to 10,000 concurrent users?

The bottleneck is mostly provider I/O and long-running agent workflows, not simple FastAPI routing. I would add Redis for shared cache, provider-specific rate limits, queues for Full-mode research jobs, and precomputation for popular tickers. Quick mode should stay low latency by returning cached price and fundamentals quickly. Full mode can stream progress or return a job ID.

At scale, I would also add authentication, user-level quotas, structured tracing, p95 latency dashboards, provider error monitoring, and deployment separation between UI, API, workers, and cache.

### 6. What's your context window strategy for long 10-K filings?

For long filings, I would not send the full 10-K to the model. I would parse the filing, split it into sections, extract relevant chunks such as business overview, risk factors, MD&A, liquidity, and legal proceedings, then summarize those chunks hierarchically. The agent should receive compact evidence snippets with citations or source section labels.

In the current code, filing-style analysis is metadata-first for BSE/NSE announcements. It does not yet parse full SEC 10-K documents. That is a roadmap item, and the context strategy would be chunking plus retrieval, not brute-force prompting.

### 7. Why did you choose FinBERT over GPT for sentiment?

FinBERT is a financial-domain classifier (`ProsusAI/finbert`), so it is a good fit for headline sentiment. It runs locally after download, which means no per-headline LLM cost, predictable labels, and easy aggregation into positive, negative, and neutral counts. GPT or Claude could explain sentiment more richly, but that is overkill for the current use case where sentiment is one numeric signal in a larger research report.

Empirical Performance & Memory Footprint (Benchmarked locally):
- **Model Load Time**: ~1.75 seconds (loaded once via singleton cache at startup)
- **Inference Latency**: ~89.1 ms per headline (~446 ms batch for 5 headlines on CPU)
- **Model Memory Footprint (Delta RSS)**: ~448.6 MB
- **Total Process Peak RSS**: ~928.4 MB

The tradeoff is nuance. FinBERT may miss sarcasm, article context, or whether news is already priced in. For FinSight, I use FinBERT for cheap classification (with a lexicon-based degraded fallback mode if FinBERT fails to load) and Claude or GPT for higher-level synthesis.

### 8. How do you evaluate the quality of the agent's output?

I would evaluate it with frozen tool fixtures and expected report properties. The report should quote the same numbers returned by tools, acknowledge missing data and `error` fields, include required sections, include the disclaimer, and avoid unsupported claims. I would measure factual consistency, error acknowledgement rate, section completion, token usage, latency, and human usefulness scores.

The current test suite covers tools, API behavior, agent error handling, and MCP wrappers. The next step is an eval harness for generated report factuality.

---

## 1. Project Pitch

### Q1. Explain FinSight in one minute.

**Answer framework:**

"FinSight is an AI-powered stock research assistant for US and Indian equities. A user enters a ticker like AAPL, TCS.NS, or RELIANCE.NS, and the app combines price technicals, fundamentals, news sentiment, corporate announcements or filings, and peer comparison into a structured research report. The architecture has a Streamlit UI, FastAPI backend, a provider-switchable LLM agent, and MCP tools for the actual financial data retrieval. I built it to learn how real multi-tool AI systems work: grounding, tool contracts, caching, failure handling, and UX for explaining financial data. It is not financial advice; it is a research assistant."

**What this answer shows:** product sense, architecture awareness, and safety.

**Follow-up trap:** If asked "Is it production ready?", do not oversell. Say it has strong project-level engineering, but production finance would need stricter data licensing, observability, auth, audit logs, evals, and compliance review.

---

### Q2. Why is this project more interesting than a normal chatbot wrapper?

**Answer framework:**

"A normal chatbot wrapper relies mostly on the model's latent knowledge. FinSight is tool-grounded. The model does not invent a stock price; it calls specific tools for price, fundamentals, sentiment, filings, and peers. The UI also exposes the tool outputs directly in tabs, so users can inspect the evidence behind the final report. The interesting engineering problem is orchestration: deciding what tools to call, normalizing different provider responses, surfacing errors honestly, and presenting uncertain financial data in a usable way."

**Learn deeper:**

- What is the difference between model knowledge and tool-grounded knowledge?
- Why is transparency important in finance UX?
- What can still hallucinate even when tools are used?

---

## 2. Architecture Deep Dive

### Q3. Walk me through what happens when a user analyses AAPL in Full mode.

**Answer framework:**

1. The user enters `AAPL` in the Streamlit sidebar and clicks `Analyse`.
2. `finsight/ui/app.py` shows a spinner and calls cached analysis logic with a 5-minute TTL.
3. The UI calls FastAPI endpoints on `localhost:8000`.
4. Quick tool calls fetch price and fundamentals directly so the UI can still show useful data even if the agent path is slow or unavailable.
5. In Full mode, the UI calls `/research`, which invokes `FinSightAgent.research()`.
6. The agent connects to the MCP server over stdio, lists available tools, sends tool schemas to Claude or GPT, executes tool calls, and collects structured results.
7. The backend returns a `ResearchReport` containing `price_data`, `fundamentals`, `sentiment`, `filing_summary`, `peer_comparison`, `final_analysis`, confidence, latency, and token usage.
8. Streamlit renders five tabs: report, price/technicals, fundamentals, news sentiment, and filing analysis.
9. Full mode also exposes a peer comparison section below the tabs.

**Follow-up trap:** If asked whether every data field always exists, say no. Provider APIs can fail or return incomplete fields, so the UI handles missing values and tool `error` fields.

---

### Q4. Why did you choose MCP instead of direct Python function calls?

**Answer framework:**

"MCP gives the tools a standard interface that an agent can discover and call. Direct Python calls would be simpler inside one process, but MCP creates a clearer boundary between agent reasoning and tool execution. In FinSight, the MCP server registers finance tools such as stock price, fundamentals, news sentiment, corporate announcements, and peer comparison. The orchestrator can list tools and call them through a protocol instead of importing every function directly. The tradeoff is more moving parts: stdio process management, schema conversion, and error handling."

**Strong extra point:**

"For a local interview project, stdio MCP is fine. In a production web system, I would likely run tool services behind HTTP or a queue, then use MCP where it adds integration value."

---

### Q5. What are the major components and their responsibilities?

**Answer framework:**

- `finsight/ui/app.py`: Streamlit dashboard, user inputs, tabs, caching, peer comparison display.
- `ui/app.py`: top-level Streamlit entrypoint wrapper.
- `finsight/api/main.py`: FastAPI app, request validation, rate limiting, CORS, research and tool endpoints.
- `finsight/agent/orchestrator.py`: provider-switchable agent loop, MCP connection, LLM tool calling, final report schema.
- `finsight/mcp_server/server.py`: MCP tool registration.
- `finsight/mcp_server/tools/price.py`: yfinance price history, RSI, MACD, moving averages, OHLCV.
- `finsight/mcp_server/tools/fundamentals.py`: valuation and financial ratios from Yahoo Finance.
- `finsight/mcp_server/tools/sentiment.py`: NewsAPI or mock headlines plus local FinBERT classification.
- `finsight/mcp_server/tools/announcements.py`: BSE/NSE corporate announcements and summary.
- `finsight/mcp_server/tools/peers.py`: concurrent peer comparison using price and fundamentals.

**Learn deeper:** Practice drawing this as a box diagram from memory.

---

### Q6. Where does caching happen and why?

**Answer framework:**

"Caching happens at multiple layers. The Streamlit UI caches analysis results for 5 minutes using `st.cache_data`, which improves demo responsiveness and avoids repeated API calls. Price and fundamentals use in-process `lru_cache` for provider data. Announcements use a time-based module cache because BSE/NSE data does not need to refresh every second. FinBERT is loaded lazily and cached as a model pipeline, because loading the model repeatedly would be expensive. The limitation is that most of this cache is process-local; in production I would use Redis or another shared cache."

**Follow-up trap:** If asked whether `lru_cache` is enough for horizontal scaling, answer no. Each process has its own cache.

---

## 3. Agent And LLM Questions

### Q7. How do you prevent hallucinated financial data?

**Answer framework:**

"I reduce hallucination by grounding the agent in tools and by preserving raw tool outputs in the response. The system prompt tells the model to use actual numbers from tools and not invent data. Each tool returns structured fields and an `error` field. The final report is only one part of the response; the UI also shows price data, fundamentals, sentiment, filings, and peer comparison directly. That means the user can inspect the underlying evidence. The remaining risk is that the model may overinterpret incomplete data, so a proper eval suite should check factual consistency between tool outputs and the generated report."

**Learn deeper:**

- Hallucination is not only "fake facts"; it can be overconfident interpretation.
- Grounding reduces risk but does not eliminate it.
- UI transparency is part of AI safety.

---

### Q8. What exactly does Claude do in this system?

**Answer framework:**

"Claude is not the data provider. It acts as the research synthesizer. It receives the user query, decides which tools to call through MCP, reads the returned JSON, and writes a structured markdown report. The actual prices, ratios, headlines, and announcements come from tools. Claude's value is in combining those signals into a readable conclusion with risks, technical picture, and fundamentals."

**Follow-up trap:** Avoid saying "Claude predicts whether the stock will go up." Say it synthesizes research signals.

---

### Q9. How would you evaluate the quality of the generated reports?

**Answer framework:**

"I would build an eval set with known tickers and frozen tool fixtures. Then I would score reports on factual consistency, missing-data honesty, structure, risk coverage, disclaimer inclusion, and latency. For example, if the fixture says P/E is 28.4, the report should not say 18. I would also include invalid tickers and provider-error cases to check that the model explains uncertainty instead of filling gaps. Finally, I would track token usage and cost per report."

**Possible metrics:**

- Factual match rate for numeric fields.
- Tool error acknowledgement rate.
- Required-section completion rate.
- Average latency and p95 latency.
- Tokens per research request.
- Human rating for usefulness and clarity.

---

### Q10. What would you do if the model calls the wrong tool?

**Answer framework:**

"I would solve it at multiple levels. First, improve tool descriptions and system instructions. Second, add orchestration rules outside the model for required tools: for example, always fetch price and fundamentals for stock analysis. Third, add validation after tool calls, so if required fields are missing, the agent can retry or degrade gracefully. For a production system, I would prefer a hybrid approach: deterministic prefetch for core data plus model-driven optional tools."

**Strong extra point:** FinSight already leans this way because the Streamlit UI fetches price and fundamentals directly before relying on the full agent response.

---

## 4. Financial Data And Modeling

### Q11. Explain RSI like you would to a non-technical interviewer.

**Answer framework:**

"RSI is a momentum indicator. It compares recent average gains and losses to estimate whether a stock may be overbought or oversold. A value above 70 is often considered overbought; below 30 is often considered oversold. FinSight computes a 14-period RSI and shows it as a gauge. It is not a buy or sell signal by itself; it is one piece of context."

**Learn deeper:** RSI can stay high during strong uptrends and low during strong downtrends, so treating it as a standalone signal is naive.

---

### Q12. Explain MACD and golden cross.

**Answer framework:**

"MACD compares short-term and longer-term exponential moving averages to understand momentum. If the MACD line is above the signal line, FinSight marks it bullish; below, bearish; close together, neutral. A golden cross is simpler: it checks whether the 50-day moving average is above the 200-day moving average. Both are trend indicators, not guarantees."

---

### Q13. Why show fundamentals and technicals together?

**Answer framework:**

"They answer different questions. Fundamentals ask whether the business looks attractive based on valuation, profitability, margins, and debt. Technicals ask how the market has recently been pricing the stock. An interviewer or investor gets a more balanced view when both are visible. For example, a company can be fundamentally strong but technically overbought, or cheap by P/E but deteriorating in momentum."

---

### Q14. What are the limitations of P/E ratio?

**Answer framework:**

"P/E is useful but incomplete. It can be distorted by one-time earnings, cyclicality, accounting differences, negative earnings, or growth expectations. A high P/E may be justified for a high-growth company, while a low P/E can be a value trap. That is why FinSight also shows P/B, ROE, debt/equity, margins, analyst rating, target price, and peer comparison."

**Learn deeper:** Practice explaining why banks, software companies, and manufacturers should not always be valued with the same metric.

---

### Q15. Why did you use FinBERT for sentiment instead of an LLM?

**Answer framework:**

"FinBERT is a financial-domain classifier. It is cheaper for repeated headline classification because it runs locally after download, and it returns consistent positive, negative, or neutral labels with confidence scores. An LLM could explain sentiment better, but it would cost more and may be less consistent for simple classification. In FinSight, sentiment is an input signal, not the final reasoning layer, so a specialized classifier fits well."

**Tradeoff:** FinBERT classifies headlines only. It may miss nuance in full articles, sarcasm, market expectations, or whether news is already priced in.

---

### Q16. How do you handle Indian stocks differently from US stocks?

**Answer framework:**

"The price and fundamentals layer can handle tickers like `TCS.NS` or `RELIANCE.NS` through Yahoo Finance. For Indian corporate events, FinSight uses BSE/NSE announcement data rather than assuming SEC filings exist. That matters because Indian companies disclose through exchanges, and the naming conventions and identifiers are different. The UI supports both US-style and NSE-style tickers."

**Follow-up trap:** Do not say every Indian ticker will work perfectly. Identifier mapping and exchange APIs can be incomplete.

---

## 5. Backend And Reliability

### Q17. What happens when a provider API fails?

**Answer framework:**

"The tools return stable payloads with default values and an `error` field instead of crashing the entire app. The API wraps errors into JSON responses, and the UI displays warnings when a section has an error. Sentiment can fall back to mock headlines when `NEWS_API_KEY` is missing. Announcements can return mock or empty structured results when BSE/NSE are unavailable. The goal is graceful degradation: users should still see what data is available."

---

### Q18. How would you scale FinSight to 10,000 concurrent users?

**Answer framework:**

"The first bottleneck would be third-party provider I/O, not Python itself. I would add a shared Redis cache for ticker data, enforce per-provider rate limits, and queue full research jobs because agent workflows can be slow. I would separate quick data endpoints from longer agent synthesis. For popular tickers, I would precompute or refresh data on a schedule. I would also add observability: request IDs, tracing across UI/API/tool calls, latency histograms, and provider error dashboards."

**Concrete scale plan:**

- 100 users: cache aggressively and set timeouts.
- 1,000 users: shared cache, worker queues, backpressure.
- 10,000 users: precomputation, horizontal API instances, distributed tracing, provider contracts.

---

### Q19. What security concerns exist in this project?

**Answer framework:**

"The main concerns are API abuse, secrets management, dependency risk, prompt injection through external text, and financial misuse. The API has basic validation and rate limiting, but production would need authentication, user-level quotas, HTTPS, secret rotation, audit logs, and stricter CORS. External news or filing text can contain prompt-injection style content, so a production agent should treat provider text as data, not instructions."

**Learn deeper:** Prompt injection is not only a chatbot problem; any retrieved external content can try to manipulate the model.

---

### Q20. How would you make the API more robust?

**Answer framework:**

"I would add typed response models for every tool endpoint, centralize provider adapters, use retries with jitter for transient errors, move caches to Redis, add structured logging, and define explicit error codes. I would also split long-running research into async jobs so the API can return a job ID and stream progress. Finally, I would add contract tests around provider payloads because yfinance and public exchange APIs can change fields."

---

## 6. Frontend And UX

### Q21. Why Streamlit instead of React?

**Answer framework:**

"For this project, speed and demoability mattered. Streamlit lets me build a useful research interface in Python without a separate frontend stack. It is good for internal tools, prototypes, and interview demos. A React app would give more control over interaction design, auth, routing, and production frontend architecture, but it would slow down iteration. If FinSight became a product, I would consider React or Next.js later."

---

### Q22. What UX decisions did you make in the Streamlit app?

**Answer framework:**

"I separated the output into five tabs because different users care about different evidence: research report, price/technicals, fundamentals, news sentiment, and filings. The sidebar keeps the main workflow simple: ticker, presets, mode, analyse. Quick mode is for fast price and fundamentals; Full mode adds the complete research workflow and peers. I also added badges, metrics, gauges, progress bars, and tables so the UI is scannable instead of being one long generated paragraph."

**Follow-up trap:** If asked about mobile, say Streamlit has limits, but the layout uses columns carefully and avoids overly dense custom CSS.

---

### Q23. Why show tool outputs separately if the report already summarizes them?

**Answer framework:**

"Because users should be able to verify the report. In finance, a polished paragraph is not enough. Showing the raw price metrics, valuation ratios, headlines, filings, and peer table makes the system more transparent and useful. It also makes errors easier to spot. If the final analysis says one thing but the metrics suggest another, the user can challenge it."

---

## 7. Testing And Quality

### Q24. What tests does the project have?

**Answer framework:**

"The tests cover the main backend and tool behavior: API health, research endpoint validation, request IDs, CORS, rate limiting, invalid tickers, price tool behavior, fundamentals, sentiment, announcements fallback and caching, peer comparison, and orchestrator behavior. Streamlit UI is not unit tested directly, but I verified that it starts and used a render smoke check for the five-tab path."

**Strong extra point:** "For a production UI, I would add Playwright smoke tests."

---

### Q25. What bug or weakness are you most aware of?

**Answer framework:**

"The biggest weakness is that external providers are unofficial or variable. yfinance, BSE/NSE endpoints, and NewsAPI can fail, rate limit, or change schemas. Another weakness is eval coverage: tool tests exist, but generated report factuality needs a dedicated benchmark. Finally, process-local caching is fine for a demo but not enough for a horizontally scaled service."

**Why this is a good answer:** It shows maturity without undermining the project.

---

### Q26. How would you debug a bad report?

**Answer framework:**

1. Check the UI tabs to see whether the underlying tool outputs are correct.
2. Call the corresponding FastAPI tool endpoint directly.
3. Inspect the MCP tool implementation if the endpoint returns bad data.
4. Check whether the agent prompt or tool descriptions caused misinterpretation.
5. Reproduce with a fixture and add a regression test.

**Learn deeper:** Always isolate whether the bug is data retrieval, normalization, orchestration, generation, or rendering.

---

## 8. Design Tradeoffs

### Q27. Why does Quick mode exist?

**Answer framework:**

"Quick mode gives users fast price and fundamentals without waiting for the entire multi-tool agent workflow. It is useful for demos and for users who only want the core snapshot. Full mode is heavier because it adds sentiment, filings, and peers. This is a UX and systems tradeoff: not every request needs maximum depth."

---

### Q28. Why not call all tools every time?

**Answer framework:**

"Calling every tool every time increases latency, cost, provider load, and failure surface. For a simple price check, news and filings may be unnecessary. FinSight currently has Full mode for richer analysis, but a more refined production system would choose tools based on intent, freshness, and cached availability."

---

### Q29. What would you improve if you had two more weeks?

**Answer framework:**

1. Add a report eval suite using frozen tool fixtures and factuality checks.
2. Move caching to Redis and add provider-level rate limiting.
3. Add streaming progress updates in the UI for long Full-mode research.
4. Add auth, user quotas, and better observability.
5. Improve filing analysis by parsing attached PDFs instead of only announcement metadata.

---

### Q30. What did you personally learn from building FinSight?

**Answer framework:**

"I learned that AI apps are mostly systems engineering. The LLM is important, but the hard parts are data contracts, provider reliability, caching, validation, graceful failure, and UX transparency. I also learned that financial AI needs humility: numbers can be stale, providers can fail, and a confident report can still be wrong unless it is grounded and inspectable."

---

## 9. Rapid-Fire Drill

Use these for quick practice. Answer each in under 30 seconds.

- **What is MCP?** A protocol that lets agents discover and call tools through a standard interface, separate from the model itself.
- **What is stdio transport?** The MCP client and server communicate through standard input and output streams rather than HTTP.
- **What does FastAPI do here?** It exposes health, research, and direct tool endpoints for the UI.
- **What does Streamlit do here?** It provides the interactive dashboard for ticker input, report rendering, metrics, and peer comparison.
- **What does FinBERT do?** It classifies financial headlines as positive, negative, or neutral.
- **What is RSI?** A momentum indicator comparing recent gains and losses, commonly interpreted with 70/30 thresholds.
- **What is MACD?** A momentum indicator comparing short and longer exponential moving averages.
- **What is P/E?** Price divided by earnings per share; a rough valuation measure.
- **What is ROE?** Return on equity; a profitability/quality metric.
- **What is debt/equity?** A leverage metric comparing debt to shareholder equity.
- **Why cache data?** To reduce latency, cost, and provider pressure.
- **Why show errors in the UI?** Because missing data should be transparent instead of silently hidden.
- **Why include a disclaimer?** The app supports research, not personalized investment advice.
- **What is graceful degradation?** Returning partial useful results when one tool or provider fails.
- **What is the biggest production risk?** Provider reliability, data licensing, and factuality of generated reports.

---

## 10. Questions To Ask The Interviewer

These make you sound thoughtful and help turn the interview into a real engineering conversation.

1. "For AI systems in your team, how do you separate deterministic business logic from model-driven reasoning?"
2. "Do you use evals or human review to measure LLM output quality?"
3. "How do you handle observability for multi-step AI workflows?"
4. "Where do you draw the line between prototype and production readiness for AI features?"
5. "What reliability expectations do you set when a product depends on third-party data providers?"
6. "How do your teams think about prompt injection from retrieved external content?"
7. "Would you prefer a deterministic workflow engine or a more autonomous agent for this kind of use case?"

---

## 11. Practice Scenarios

### Scenario A: The interviewer challenges data accuracy.

**Say:**

"That is exactly the right concern. FinSight reduces risk by grounding reports in tool outputs and showing those outputs separately. But I would not claim perfect accuracy. The next step would be frozen fixtures and factuality evals that compare generated text against source JSON."

### Scenario B: The interviewer says MCP is overengineering.

**Say:**

"For a tiny app, direct function calls are simpler. I used MCP because the learning goal was multi-tool agent architecture and clean separation between tool server and agent. I can explain the tradeoff: MCP adds process and transport complexity, but it creates a standardized tool boundary."

### Scenario C: The interviewer asks why not just use ChatGPT browsing.

**Say:**

"Browsing is general purpose. FinSight has structured data contracts and domain-specific tools. That lets the UI show exact fields like RSI, P/E, ROE, sentiment score, and peer rankings. Structured outputs are easier to test and safer to reuse than free-form browsing results."

### Scenario D: The interviewer asks if this gives investment advice.

**Say:**

"No. It is a research assistant. It summarizes public data and signals, includes a disclaimer, and should not be used as personalized financial advice. A regulated product would need suitability checks, compliance review, audit trails, and licensed data."

---

## 12. Debugging Case Studies

Use these as real examples of production-style debugging: identify the layer, verify the assumption, make the failure observable, then fix the smallest useful thing.

### Case Study A: NewsAPI key configured but app says it is missing

**Symptom:** The UI showed: `NEWS_API_KEY is not set. Returning mock sentiment data.`

**Root cause:** The `.env` file contained the variable name, but the value was empty: `NEWS_API_KEY=`. In another path, the sentiment tool could also be imported directly without loading `.env`, so relying only on the orchestrator to call `load_dotenv()` was fragile.

**Debugging steps:**

- Checked the exact environment variable name expected by the code: `NEWS_API_KEY`.
- Parsed `.env` with `dotenv_values()` and printed only presence, length, and non-empty status, not the secret.
- Verified that the error was raised before any NewsAPI request, so it was not an Indian-news coverage problem.

**Fix:** Put the actual key in `.env`, restart the backend, and load the root `.env` from the sentiment module as well. Also rotate any key that was accidentally pasted into logs or chat.

**Interview framing:** "I did not assume the third-party API was wrong. I first verified whether the process could actually see the secret. The failure was configuration, not provider coverage."

### Case Study B: Hugging Face 500 while loading FinBERT

**Symptom:** A traceback showed `500 Internal Server Error` from `huggingface.co/api/models/ProsusAI/finbert/discussions`.

**Root cause:** This was not NewsAPI. It came from Transformers while loading the local FinBERT sentiment model. The loader tried to access Hugging Face model metadata for safetensors conversion, and Hugging Face returned a server-side 500.

**Debugging steps:**

- Followed the traceback to the failing external service and endpoint.
- Separated the news-fetching layer from the sentiment-classification layer.
- Confirmed the app used `pipeline("sentiment-analysis", model="ProsusAI/finbert")`.

**Fix:** Load FinBERT with normal PyTorch weights by passing `model_kwargs={"use_safetensors": False}`. This avoids the extra safetensors conversion metadata call while preserving local FinBERT inference.

**Interview framing:** "The API key was not involved. The failure was in the ML dependency path, so I reduced dependence on a flaky metadata call and kept the sentiment model local."

### Case Study C: Local 429 on `/tools/filings`

**Symptom:** The UI or API returned `429 Client Error: Too Many Requests for url: http://localhost:8000/tools/filings`.

**Root cause:** This was the app's own FastAPI rate limiter, not BSE/NSE or NewsAPI. The middleware allowed 10 requests per 60 seconds per client IP. A full Streamlit analysis can call multiple endpoints quickly, and repeated refreshes can hit the local limit.

**Debugging steps:**

- Checked the failing URL: it was `localhost`, so the rejection came from our backend.
- Found `RateLimitMiddleware` in `finsight/api/main.py`.
- Compared the UI flow with the limit: full analysis can call research plus direct tool endpoints, so the demo can exceed 10 requests/minute.

**Fix:** Keep the default test-friendly limit at 10 requests per 60 seconds, but make it configurable with `FINSIGHT_RATE_LIMIT_REQUESTS` and `FINSIGHT_RATE_LIMIT_WINDOW_SECONDS`. Return a `Retry-After` header on 429 so clients can back off cleanly.

**How to handle locally:** For demos, set a higher limit in `.env`, such as:

```env
FINSIGHT_RATE_LIMIT_REQUESTS=60
FINSIGHT_RATE_LIMIT_WINDOW_SECONDS=60
```

Then restart the backend. In production, rate limits should usually be user/API-key based, backed by Redis, and coordinated with provider-specific limits.

**Interview framing:** "A 429 is not always from the external provider. I checked where the URL pointed, found it was our local API, and made the limit configurable instead of removing protection entirely."

### Case Study D: `TaskGroup` error in the research report

**Symptom:** The report section showed: `Error occurred: unhandled errors in a TaskGroup (1 sub-exception)`.

**Root cause:** The MCP client was launching the server by file path: `finsight/mcp_server/server.py`. When Python runs a script by path, it puts that script's directory on `sys.path`, not necessarily the project root. The MCP subprocess crashed with `ModuleNotFoundError: No module named 'finsight'`, and the async MCP wrapper surfaced that subprocess crash as a generic `TaskGroup` error.

**Debugging steps:**

- Reproduced the failure locally and captured the exact subprocess stderr.
- Verified that the parent process could import `finsight` while the spawned subprocess could not.
- Confirmed the path difference by printing `sys.path` inside the subprocess.

**Fix:** Start the MCP server with the project root on `PYTHONPATH`, not just the server file directory. I also made the path construction explicit in `FinSightAgent._mcp_server_params()` so the subprocess always runs with the same root context.

**Interview framing:** "This was a packaging/path issue rather than an LLM issue. The agent looked broken because its subprocess couldn't import its own package. I fixed it by making the subprocess environment deterministic."

### Case Study E: Parallel tool hydration with background research

**Symptom:** In `Full` mode the app showed a spinner for a long time, even though price, fundamentals, and filings were available quickly. Users could not see any evidence until the full `/research` call completed.

**Root cause:** The UI treated the agent synthesis call as the only output. It blocked on `/research`, which included tool execution plus LLM synthesis, so the faster direct tool endpoints were not surfaced independently.

**Debugging steps:**

- Identified that `finsight/ui/app.py` already had direct tool endpoints like `/tools/price` and `/tools/fundamentals`.
- Verified that these endpoints were much faster than `/research` during normal operation.
- Changed the UI workflow to fetch direct tool output in parallel and render it immediately.
- Launched `/research` in a background thread and kept a `research_status` flag in session state so the app could show partial evidence and later refresh the final report.

**Fix:** Separate the fast hydration layer from the slow synthesis layer. The UI now:

- fetches price and fundamentals concurrently,
- also fetches sentiment and filings in parallel,
- displays those results immediately,
- starts `/research` in the background,
- and updates the final report when the background task finishes.

**Impact:** The app feels much more responsive. Users get evidence quickly instead of staring at a spinner, and the full LLM-generated analysis still arrives when it is ready. This reduces perceived latency and keeps the system usable when the agent takes longer to complete.

---

## 13. Red Flags To Avoid

- Do not say "the AI knows the stock price." The tools retrieve the stock price.
- Do not quote benchmark numbers unless you have actually measured them.
- Do not claim the app is production-ready for regulated finance.
- Do not hide provider failures; explain graceful degradation.
- Do not call technical indicators predictions.
- Do not say Streamlit is always better than React; explain why it fits this project.
- Do not say MCP is always necessary; explain the tradeoff.
- Do not describe the agent as magic; describe the loop: prompt, tool schemas, tool calls, JSON results, synthesis.

---

## 14. Final Memorized Close

"The main thing I learned from FinSight is that useful AI products are not just prompts. They are systems: data contracts, tools, caching, error handling, UI transparency, and evals. The LLM is the synthesis layer, but the engineering around it is what makes the output trustworthy."

---
## RAG — Step 1: PDF Extraction and Chunking

### What I built
- finsight/rag/chunker.py
- extract_text_from_pdf(): pdfplumber page-by-page extraction
- chunk_document(): sentence-aware sliding window with overlap

### The decision I can defend: chunk_size=512, overlap=64

Why 512 tokens:
- Too small (128): a single financial paragraph gets split across
  multiple chunks, losing context. "Revenue grew 18% YoY driven by..."
  gets cut before the driver is named.
- Too large (1024+): retrieval becomes imprecise. A 1024-token chunk
  might contain both the risk section and the growth section —
  asking about risks retrieves irrelevant growth text too.
- 512 is the community standard for dense retrieval and matches
  the max sequence length of most sentence-transformer models.

Why 64-token overlap:
- Without overlap: a key sentence that falls at a chunk boundary
  appears in neither chunk cleanly. The embedding captures half
  the sentence, retrieval misses it.
- With overlap: the boundary sentence appears fully in at least
  one chunk. 64 tokens (~256 chars) is typically 2-3 sentences —
  enough to capture any boundary sentence without doubling storage.

Why sentence-aware (not character-based):
- Splitting at exactly 512*4=2048 chars cuts mid-sentence.
- Mid-sentence chunks embed poorly — the vector represents an
  incomplete thought.
- Sentence-aware splitting ensures each chunk is a coherent unit.

### Question I can answer cold
"Why not just split every 500 characters?"
"Character splitting cuts mid-sentence. The embedding then represents
a fragment, not a thought. When you query 'what did management say
about margins', a fragment embedding matches poorly. Sentence-aware
splitting ensures every chunk is semantically complete, which
directly improves retrieval precision."

### What I would do differently in production
- Use a proper tokenizer (tiktoken or the model's own tokenizer)
  instead of len(text)//4. The 1 token = 4 chars approximation
  breaks on Hindi/regional text in Indian annual reports.
- Detect and preserve tables separately — pdfplumber has
  extract_tables() which keeps table structure intact.
  Text-chunked tables lose their row/column relationships.

---
## RAG — Step 2: Embeddings

### What I built
- finsight/rag/embedder.py
- Embedder class with lazy loading and module-level singleton
- all-MiniLM-L6-v2: 384 dimensions, CPU-only, ~80MB

### The concept: what an embedding actually is
An embedding converts text into a point in high-dimensional space
such that semantically similar texts land near each other.
"Revenue increased" and "Sales grew" end up close together.
"It rained in Mumbai" ends up far from both.
Distance is measured by cosine similarity:
  cos(θ) = dot(A,B) / (|A| * |B|)
Range: -1 (opposite) to +1 (identical direction).

### The decision I can defend: why not OpenAI embeddings

OpenAI text-embedding-3-small costs $0.02 per million tokens.
For a 300-page annual report (~150,000 tokens), that is $0.003
per ingestion — negligible individually.

But the real reason to use a local model:
1. Latency: local inference has no network round trip
2. Privacy: annual reports can be price-sensitive documents
3. Portfolio signal: shows I can run ML inference locally,
   not just call APIs

The tradeoff: all-MiniLM-L6-v2 was not trained on financial text.
Terms like "EBITDA", "PAT", "QoQ" are rare in its training data.
This is why I add BM25 hybrid search in Step 4 — keywords
that embed poorly get caught by exact-term matching.

### The critical mistake to avoid: asymmetric models
Query and documents MUST use the same embedding model.
If you embed documents with model A and queries with model B,
cosine similarity is meaningless — the vector spaces are different.
In embedder.py, embed_query() and embed_texts() call the same
self._model instance. This is not an accident.

### Question I can answer cold
"What is cosine similarity and why use it over Euclidean distance?"
"Cosine similarity measures the angle between two vectors, ignoring
magnitude. For text embeddings, two sentences with the same meaning
should point in the same direction regardless of sentence length.
A short sentence and a long sentence expressing the same idea will
have different magnitudes but nearly the same angle. Euclidean
distance conflates direction and magnitude, so it penalises length
differences incorrectly."

---
## RAG — Step 3: Vector Store (ChromaDB)

### What I built
- finsight/rag/vector_store.py
- ChromaDB PersistentClient, one collection per document
- Cosine similarity space, metadata preserved per chunk

### The concept: how vector search actually works
ChromaDB uses HNSW (Hierarchical Navigable Small World) graphs
for approximate nearest neighbour search.

HNSW builds a multi-layer graph where:
- Bottom layer: all vectors connected to close neighbours
- Upper layers: progressively fewer nodes, longer-range connections
- Search: enter at top layer, greedily descend to nearest neighbour

Why approximate and not exact:
Exact nearest neighbour in 384 dimensions requires comparing your
query against every stored vector — O(n * d). With 50,000 chunks
from a large annual report, that is 19.2 million multiplications
per query. HNSW finds the true nearest neighbour ~99% of the time
in O(log n) comparisons.

### The decision I can defend: one collection per document
Alternative: one global collection for all documents, filter by
metadata (doc_id == "tcs_annual_2024").

Why per-document collections instead:
1. Query isolation: no cross-document contamination possible
2. Clean deletion: delete_document() drops the collection,
   no leftover chunks from other documents
3. Chunk count accuracy: collection.count() always refers to
   exactly one document

The tradeoff: you cannot do cross-document retrieval
("find all mentions of margin pressure across all ingested reports").
For FinSight's use case (one document at a time Q&A),
this is the right call.

### The bug I avoided: n_results > collection.count()
ChromaDB raises ValueError if you request more results than
exist in the collection. My query() method caps n_results at
min(k, collection.count()). Without this, querying a collection
with 3 chunks and k=5 crashes. Always handle this edge case.

### Question I can answer cold
"What is HNSW and why is it used in vector databases?"
"HNSW is an approximate nearest neighbour algorithm that organises
vectors into a multi-layer graph. You enter the search at the top
layer with few nodes and long-range connections, greedily descend
to your nearest neighbour, then refine at the bottom layer with
all vectors. It finds the true nearest neighbour ~99% of the time
in O(log n) vs O(n) for exact search. At 100,000 vectors with 384
dimensions, exact search requires 38.4 million multiplications per
query. HNSW does it in roughly 17 comparisons."

---
## RAG — Step 4: Hybrid Retrieval (Dense + BM25 + RRF)

### What I built
- finsight/rag/retriever.py
- HybridRetriever: ChromaDB dense + BM25 sparse, fused with RRF
- Default weights: dense=0.7, sparse=0.3

### The concept: why hybrid search

Dense retrieval (embeddings) captures meaning.
"Revenue grew strongly" matches "sales increased significantly"
even though no words overlap. This is the superpower of embeddings.

BM25 captures exact terms.
BM25 score for a document = sum of:
  IDF(term) * (TF * (k1+1)) / (TF + k1 * (1 - b + b * doc_len/avgdl))
where k1=1.5, b=0.75 are standard constants.
IDF is high for rare terms. "EBITDA" appears in few chunks —
high IDF. "the" appears everywhere — low IDF, ignored.

Weakness of dense alone:
"₹4,200 crore revenue" embeds similarly to "₹4,800 crore revenue"
because the sentence structure is identical. If you ask
"what was the exact revenue figure", dense retrieval cannot
distinguish them. BM25 on "4,200" finds the exact match.

### The concept: RRF (Reciprocal Rank Fusion)

Problem: dense and sparse return different scores on different scales.
Dense similarity: 0.0 to 1.0.
BM25 score: 0.0 to ~15.0 depending on corpus.
You cannot average them directly.

RRF converts ranks (not scores) into a combined score:
  score(chunk) = Σ weight_i * (1 / (rank_i + 60))

The constant 60 is the RRF constant (from the original 2009 paper).
It controls how much rank-1 dominates rank-2:
  rank-1: 1/(1+60) = 0.0164
  rank-2: 1/(2+60) = 0.0161
  rank-10: 1/(10+60) = 0.0143
The gap between ranks is small — a rank-1 in one method
and rank-3 in another fuses well. Without the constant (using
just 1/rank), rank-1 would dominate too heavily.

### The decision I can defend: dense_weight=0.7

Financial documents are mostly narrative with some specific data.
"Management expects margins to improve driven by automation" → dense.
"EBITDA margin 24.3% vs 21.8% last quarter" → sparse.
0.7/0.3 reflects that ratio. If the document were a data sheet
full of numbers, I would flip to 0.3/0.7.

### Question I can answer cold
"What is BM25 and how does it differ from TF-IDF?"
"Both weight terms by frequency and rarity. TF-IDF simply
multiplies TF * IDF. BM25 adds two improvements: it saturates
term frequency (adding the 20th occurrence of a word matters
less than the 1st — TF-IDF treats them linearly), and it
normalises by document length (long documents get penalised
for having high raw TF just because they are long). These
make BM25 consistently better than TF-IDF for retrieval,
which is why it is still the default in Elasticsearch."

---
## RAG — Step 5: Generator and Pipeline

### What I built
- finsight/rag/generator.py — LLM grounded answer generation
- finsight/rag/pipeline.py — full orchestration: ingest + query

### The concept: why RAG prevents hallucination (and its limits)

RAG reduces hallucination because the LLM's context window
contains the actual source text. Claude is instructed to only
use the provided excerpts. If "₹9,000 crore" is in the chunk,
the answer contains "₹9,000 crore". If it is not, the system
prompt forces "I could not find this information."

The limit: RAG prevents hallucination about facts IN the document.
It does not prevent the LLM from:
- Misquoting a number (reading "9,000" as "90,000")
- Confidently stating something from a low-quality chunk

This is why retrieval quality matters so much — a wrong chunk
leads to a wrong answer even with a perfect system prompt.

### The decision I can defend: "I could not find" over silence

Some RAG systems return an empty answer when retrieval fails.
I instruct the model to say explicitly "I could not find this
information in the uploaded document."

Why: a financial analyst needs to know whether the absence of
an answer means (a) the model failed or (b) the document
genuinely does not contain that information. Explicit "not found"
gives that signal. Silence is ambiguous.

### The decision I can defend: batch size 32 for ingestion

Embedding 32 chunks at once vs 1 at a time:
- 1 at a time: model overhead per call dominates. 1000 chunks
  = 1000 overhead calls.
- 32 at once: model processes in one forward pass,
  GPU/CPU batching applies. ~10-15x faster in practice.
- Too large (512+): exceeds memory on CPU inference.
32 is the standard default for sentence-transformers on CPU.

### The hard problem: BM25 index survives only in memory

ChromaDB persists vectors to disk. BM25 index does not —
it is rebuilt in memory from the chunk corpus.

If the server restarts, the BM25 index is gone.
My pipeline.query() handles this: if doc_id not in
self._chunk_cache, it re-fetches all chunks from ChromaDB
and rebuilds the BM25 index before retrieval.

This is a real production concern. The production solution
is to serialise the BM25 index to disk with pickle alongside
the ChromaDB directory, or move to a dedicated search service
like Elasticsearch that persists both.

### Question I can answer cold
"How does your RAG system prevent hallucination?"
"Two ways. First, the system prompt explicitly instructs Claude
to only use the provided excerpts and to say 'I could not find
this information' if the answer is absent — it cannot use
outside knowledge. Second, every answer must include [Page X]
citations. If Claude cannot cite a page, it cannot make the
claim. The remaining risk is retrieval failure — if the wrong
chunk is retrieved, Claude answers correctly from the wrong
context. That is why retrieval quality (hybrid search, good
chunking) is more important than prompt engineering for RAG."

---
## RAG — Step 6: API and UI. Full system complete.

### What I built
- POST /rag/ingest — PDF upload, validation, pipeline ingestion
- POST /rag/query — grounded Q&A with sources
- GET/DELETE /rag/documents — document management
- Streamlit Tab 6: upload, query, source citations, manage

### The full RAG pipeline — what I can walk through cold

"A user uploads a Reliance FY2024 annual report PDF.

The API validates it: must be PDF, under 20MB, alphanumeric doc_id.
It saves to a temp file, calls rag_pipeline.ingest(), then deletes
the temp file.

Ingestion: pdfplumber extracts text page by page, preserving page
numbers. The chunker splits into 512-token sentence-aware windows
with 64-token overlap. Each chunk gets metadata: chunk_id, doc_id,
page_start, page_end, char offsets, token estimate.

The embedder encodes chunks in batches of 32 using all-MiniLM-L6-v2
(384 dims, local CPU). Vectors go into ChromaDB in a dedicated
collection for this doc_id. A BM25 index is built in memory from
the same chunk corpus.

When the user asks 'What did management say about margin pressure':

The query is embedded with the same model. ChromaDB returns top 10
chunks by cosine similarity. BM25 scores all chunks for the query
terms. RRF fuses both ranked lists (dense_weight=0.7, sparse=0.3),
returns top 5 by fused score.

The 5 chunks go into Claude's context with a strict system prompt:
answer only from excerpts, cite [Page X] for every claim, say
'could not find' if absent. Claude returns a cited answer.

The Streamlit UI shows the answer in an info box and the sources
in an expandable section with page ranges and similarity scores."

### What I would add with more time (answer this in every interview)

1. Cross-encoder re-ranking
   After retrieving top 20 chunks, run a cross-encoder
   (ms-marco-MiniLM-L-6-v2) to re-score all 20 and pick top 5.
   Cross-encoders compare query and chunk jointly — more accurate
   than bi-encoder cosine similarity, but too slow to run on all chunks.
   The two-stage approach (bi-encoder retrieve, cross-encoder re-rank)
   is industry standard for production RAG.

2. Table extraction
   pdfplumber has extract_tables(). Annual reports have P&L tables,
   balance sheets. Text-chunked tables lose row/column relationships.
   "Revenue: 9,000 | Expenses: 6,200 | PAT: 2,800" should stay
   as a unit, not be split across chunks.

3. Eval harness
   Build 20 question-answer pairs from a known document.
   Measure retrieval Recall@5: what fraction of correct answers
   appear in the top 5 retrieved chunks.
   Measure answer accuracy: does the final answer contain the
   correct fact.
   Right now I have no number for retrieval quality. That number
   is what you optimise chunking and retrieval weights against.

4. Persistent BM25
   Serialise BM25 index to disk with pickle on ingest.
   Load from disk on server restart.
   Current implementation rebuilds from ChromaDB on restart —
   correct but slow for large documents.

### One-line RAG pitch for interviews
"I built a document Q&A feature that lets users upload any annual
report PDF and ask questions against it — grounded answers with
page citations, using hybrid dense+BM25 retrieval fused with RRF,
without LangChain, so I understand every component."
