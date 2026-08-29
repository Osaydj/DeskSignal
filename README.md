# DeskSignal — AI Business Analytics Assistant

DeskSignal is a portfolio and educational business analytics application built
with Python, Streamlit, Pandas, Plotly, and the Anthropic Python SDK. Python
calculates every statistic. Claude optionally explains those calculated results
and can retrieve relevant sections from a small, entirely fictional company
knowledge base.

The repository runs in **analytics-only mode by default**. CSV validation,
filters, KPIs, charts, grouped tables, and downloads require no API key and make
no external AI request. Claude features remain in the codebase but must be
explicitly enabled by the person operating the application.
## Live Demo

[Launch the DeskSignal analytics dashboard](https://desksignal-business-analytics.streamlit.app)

The public demo runs in analytics-only mode using fictional sales data. AI features are disabled, and no API key is required.
## Version 2 — Released

Version 2 is complete, tested, and deployed from the `main` branch. The public demo now runs Version 2 in analytics-only mode. AI features remain disabled by default and require explicit operator configuration.

### Version 2 Features

- Seven-section dashboard navigation: Overview, Trends, Anomalies, Forecasting, Products & Markets, Data Explorer, and AI Insights
- Deterministic daily revenue and profit anomaly detection
- Rolling historical baselines and anomaly scores
- Interactive anomaly charts with flagged spikes and drops
- Detailed anomaly investigation tables
- Deterministic daily revenue and profit forecasting
- Forecast horizons of 7, 14, 30, 60, or 90 days
- Historical lookback windows of 30, 60, 90, 180, or 365 days
- Day-of-week patterns, recent linear trends, and statistical uncertainty ranges
- Warnings when the forecast horizon exceeds the historical lookback
- Forecast tables and CSV downloads
- Spreadsheet-safe CSV downloads that neutralize formula-like text cells
- 81 automated tests

The anomaly detection and forecasting features are calculated entirely with Python. They do not use an AI model or make an external API request.

### Forecasting Limitations

Forecasts are estimates based on recent historical patterns, not guaranteed future results. Outliers, limited history, seasonal changes, and unexpected business events can affect accuracy. A short historical lookback may overreact to temporary changes, especially when forecasting a longer period.

Negative profit forecasts are possible when the recent profit trend is declining. Revenue forecasts are prevented from falling below zero. Forecasts should be used as decision-support signals rather than confirmed business outcomes.

An anomaly flag is a lead for investigation, not proof of a business problem.
## Version 1 Features

- CSV upload and required-column validation
- Date, category, and region filters
- Revenue, profit, margin, transaction, unit, and order-value KPIs
- Best-performing product and region by revenue
- Interactive revenue and profit trend charts
- Category, region, and product performance charts
- Friendly validation errors
- Financial arithmetic checks for revenue and profit
- A 10 MB upload limit and a 100,000-row processing limit
- Reusable calculation and visualization functions
- Accessible chart-equivalent tables and filtered CSV downloads
- Button-driven AI Executive Summary using Claude
- Evidence IDs that keep Python-calculated facts separate from interpretations
- Filter-aware session state that prevents stale summaries
- Ask Your Data chat with seven approved read-only analytics tools
- Beginner-friendly RAG with one approved local document-search tool
- Four fictional policy, product, and strategy documents
- TF-IDF retrieval that sends only the highest-scoring sections to Claude
- Separate analytics and company-document evidence in every hybrid answer
- Button-driven AI Recommendations with three to five potential management actions
- Structured observation, supporting data, action, reasoning, and limitations
- Application-side rejection of invented citations and unsupported statistics
- Strict tool schemas plus application-side Pydantic validation
- Filter-aware conversation history
- Bounded tool-use loop with no arbitrary code execution
- Friendly API configuration, service, and malformed-response errors
- Automated tests

## Project Structure

```text
DeskSignal/
├── knowledge_base/      # Fictional policy, product, and strategy Markdown
├── desksignal/
│   ├── analytics/       # KPI, filtering, and aggregation functions
│   ├── ai/              # Claude client, prompt, context, and response models
│   ├── data/            # CSV loading, schema, and validation
│   ├── rag/             # Document loading, chunking, and local retrieval
│   ├── ui/              # Streamlit dashboard layout
│   └── visualization/   # Plotly chart builders
├── sample_data/         # Entirely fictional test data
├── scripts/             # Sample-data generator
├── tests/               # Automated tests
├── streamlit_app.py     # Streamlit entry point
└── requirements.txt
```

## Required CSV Columns

```text
transaction_id
date
product
category
region
customer_type
units_sold
unit_price
revenue
cost
profit
```

Column names are normalized before validation. For example, `Unit Price` is
accepted as `unit_price`. Extra columns are preserved.

Uploaded files must also follow these rules:

- Use ISO dates in `YYYY-MM-DD` format. ISO timestamps are normalized safely.
- `units_sold` must be a positive whole number.
- `unit_price` must be positive; `revenue` and `cost` cannot be negative.
- `revenue` must equal `units_sold × unit_price` within one cent.
- `profit` must equal `revenue − cost` within one cent.
- Files can be at most 10 MB and contain at most 100,000 rows.

## Requirements

- Python 3.12
- Git, when cloning the repository
- A modern web browser
- VS Code is recommended but not required

## Clone or Download

Copy the repository's HTTPS URL from GitHub, then run:

```bash
git clone https://github.com/Osaydj/DeskSignal.git
cd DeskSignal
```

Alternatively, download the repository ZIP from GitHub, extract it, and open
the folder containing `streamlit_app.py`.

## Run Locally on Windows

Create and activate a virtual environment:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, allow it only for the current terminal and
activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

Install the application and testing packages:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Run the tests and start Streamlit:

```powershell
python -m pytest
python -m streamlit run streamlit_app.py
```

## Run Locally on macOS or Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pytest
python -m streamlit run streamlit_app.py
```

Open `http://localhost:8501` if the browser does not open automatically.

## Configure the Claude API Securely

AI is optional. The dashboard, deterministic charts, document loading, and local
retrieval work without an API key. To enable the Executive Summary,
Recommendations, and Ask Your Data features, copy the safe example file:

```powershell
Copy-Item .env.example .env
```

Open `.env` in VS Code and replace the blank value locally:

```dotenv
AI_FEATURES_ENABLED=true
ANTHROPIC_API_KEY=your-real-key-goes-here
CLAUDE_MODEL=claude-sonnet-5
```

Restart Streamlit after changing `.env`. The repository's `.gitignore` excludes
`.env` and `.streamlit/secrets.toml`; `.env.example` contains no secret and is
safe to commit. You can verify the protection before your first commit:

```powershell
git check-ignore -v .env
git status
```

Never paste a real key into Python, README files, screenshots, issues, commits,
or `.env.example`. If a key is ever committed, revoke it in the Anthropic Console
and create a replacement; removing it from the latest commit is not enough.

Do not enable a shared server-side API key in an unauthenticated public
deployment. A public production version needs authentication, call quotas,
rate limiting, monitoring, and a spending limit. For a source-code portfolio,
each developer should configure their own local key.

Click **Generate AI Executive Summary** only when you want to make an API call.
The application sends KPI, monthly trend, category, product, and region
aggregates. It does not send the uploaded transaction rows. API calls can incur
Anthropic usage charges.

The **Ask Your Data** box also makes API calls only after you submit a question.
Its answers always use the currently filtered dashboard data. Changing a date,
category, or region filter clears the conversation so an old answer is never
shown as if it belongs to a new data scope.

The **AI Recommendations** section makes an API call only when you click its
button. Recommendations use calculated aggregates and at most five locally
retrieved company sections. Uploaded transaction rows are not sent to Claude.

## How the AI Grounding Works

1. Pandas calculates KPIs, monthly changes, performance breakdowns, and a
   transparent month-over-month review rule.
2. Python converts those results into a compact JSON evidence catalog.
3. Claude selects evidence IDs and supplies cautious, number-free
   interpretations in a validated structured response.
4. Streamlit renders each fact from Python and shows Claude's interpretation
   separately. Unknown evidence IDs or numeric claims in Claude-authored text
   are rejected as malformed.

The unusual-change rule flags absolute monthly revenue or profit changes of at
least 20%. It is an investigation cue, not a statistical anomaly detector.

## How Ask Your Data and RAG Work

DeskSignal does not send the entire CSV to Claude. It sends the user's question,
the current filtered date range, a compact list of available labels, basic
knowledge-base metadata, and eight strict tool definitions:

```text
calculate_revenue
calculate_profit
compare_periods
analyze_product_performance
analyze_region_performance
analyze_category_performance
get_time_series
search_company_knowledge
```

Claude selects a tool and supplies structured arguments. The application checks
the name against a fixed whitelist and validates all arguments with Pydantic.
Python/Pandas then performs the calculation and returns aggregate JSON with
evidence IDs. Claude may request another calculation before returning its final
structured explanation.

For a hybrid question, Claude can first request a Pandas calculation and then
request a focused knowledge-base search. The final response has two citation
lists, so Streamlit renders these separately:

- **Analytics evidence:** exact facts calculated by Python/Pandas
- **Company document evidence:** retrieved section title, source path, and excerpt

The four documents describe the fictional company **Northstar Office Co.** They
contain sales and discount policies, inventory and regional rules, product notes,
and a 2025 commercial strategy. They are written to align with patterns in the
fictional sample data, but the application still labels timing as an association
rather than proof of causation.

## Beginner-Friendly RAG Pipeline

RAG means **Retrieval-Augmented Generation**. In this version, generation is
"augmented" by searching for relevant company text before Claude writes an
answer:

1. **Load:** Python reads only `.md` files under the approved `policies`,
   `products`, and `strategy` folders.
2. **Chunk:** Each Markdown heading becomes a small section of at most 1,000
   characters, with a title, type, source path, and stable chunk ID.
3. **Index locally:** The retriever tokenizes sections and builds TF-IDF weights.
   TF-IDF gives more importance to words that are distinctive within this small
   collection.
4. **Retrieve:** `search_company_knowledge` compares the question with every
   section using cosine similarity and returns only the top one to five sections
   with a score above zero.
5. **Combine:** For a hybrid question, Claude receives compact analytics tool
   results and only the retrieved document sections—not raw CSV rows or every
   document.
6. **Ground:** Claude must cite evidence IDs from the two separate source types.
   The application rejects unknown IDs and numeric claims absent from cited
   evidence.
7. **Display:** Streamlit shows the calculated analytics facts, retrieved company
   sections, Claude's explanation, and limitations under separate labels.

TF-IDF is a good first RAG implementation because it is transparent, fast, and
adds no dependency or external embedding bill. Its limitation is that it mostly
matches words rather than meaning. A later version can swap the retriever for
embeddings and a vector store without changing the tool-use or evidence model.

## How AI Recommendations Work

This feature is decision support, not an automatic decision maker:

1. Python builds the same deterministic KPI, trend, product, category, and
   regional evidence used elsewhere in DeskSignal.
2. Python creates focused retrieval queries from measured strongest, weakest,
   margin, region, and unusual-change patterns.
3. Local TF-IDF retrieval selects at most five relevant company sections. It
   does not send every document.
4. Claude receives only the calculated analytics package and retrieved sections.
5. A strict Pydantic response model requires three to five recommendations. Each
   recommendation includes an observation, analytics evidence IDs, optional
   document evidence IDs, a potential action, reasoning, confidence, and a
   limitation.
6. Python rejects unknown evidence IDs, duplicate recommendations, document
   context that was completely ignored, and numbers written directly by Claude.
7. Streamlit resolves every analytics ID back to the exact Python-calculated fact
   and displays every cited document excerpt with its source path.

Suggested actions are framed as investigations, reviews, comparisons, tests,
monitoring, or reversible pilots. A policy or promotion that lines up with a
sales change is treated as a possible lead, never as proven causation.

For the fictional sample data, useful recommendation themes may include reviewing
the leading product's repeatable drivers, protecting Office Furniture margins,
examining regional allocation, or testing product-bundle ideas. Claude must still
support each theme with the actual evidence supplied during that request.

The loop allows at most six tool calls. Tool filters use exact Pandas comparisons;
there is no `eval`, `exec`, generated SQL, shell access, arbitrary file access,
or code-execution tool. Final evidence IDs must exist, and numeric text is
rejected when it does not appear in the cited analytics or document evidence.

For questions such as "Why did revenue decline?", DeskSignal can identify measured
contributors by product, region, or category. It cannot prove an external cause,
so the answer must state that limitation.

Example questions:

```text
Which region generated the most profit?
Why did revenue decline in July?
What were our three strongest products?
Compare Q1 and Q2.
Which category has the lowest margins?
HD Webcam sales dropped in July. Is there anything in our company documents that might explain this?
```

The hybrid example should reveal that the fictional data shows a sharp June-to-
July decline for HD Webcam revenue, while the fictional product notes say a
promotion ended and featured placement moved to Portable Monitor. That is a
plausible investigation lead, not proof that the document events caused the
change.

## Run the Tests

Install the development packages:

```powershell
python -m pip install -r requirements-dev.txt
```

Run:

```powershell
python -m pytest
```

The GitHub Actions workflow runs the same tests, bytecode compilation, and
dependency consistency check for every push and pull request.

No new package is required for RAG. The TF-IDF and cosine-similarity code uses
Python's standard library; the existing packages in `requirements.txt` are
unchanged.

## Generate Fresh Fictional Data

```powershell
python scripts/generate_sample_data.py
```

The default command creates `sample_data/sample_sales_data.csv` with 5,000
fictional transactions and a fixed seed for reproducibility.

## Data Safety

Use only fictional, generated, or public sample data. Never commit `.env`, API
keys, private company data, or uploaded customer data to GitHub.

## Portfolio Deployment Mode

Keep this setting for a free public analytics demonstration:

```dotenv
AI_FEATURES_ENABLED=false
```

This mode displays the complete deterministic dashboard and a clear notice that
the optional Claude features are disabled. It does not require an Anthropic
account and cannot spend API credits.


## Author

Created by [Osayd Jahanzeb](https://github.com/Osaydj).

## Copyright

Copyright © 2026 Osayd Jahanzeb. All rights reserved.

This source code is provided for portfolio review only. No license is granted to copy, modify, redistribute, sublicense, or sell this software.
