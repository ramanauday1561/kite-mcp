# NIFTY 50 F&O Strategy Builder

An adaptive options strategy builder for the NIFTY 50 index. It analyses the
market every trading day, publishes a **CALL / PUT / stand-aside** verdict with
a concrete options trade plan, then checks its own predictions against what the
market actually did and **reweights itself accordingly**.

Everything runs on free infrastructure: GitHub Actions for compute, GitHub Pages
for hosting, and free market-data endpoints. There is no server and nothing to pay for.

> **Not investment advice.** This is an educational research project. F&O trading
> carries a high risk of loss. Signals are model estimates with no guarantee of
> accuracy. Verify independently and never risk money you cannot afford to lose.

---

## How it works

```
free data sources        13 strategy voters + LLM panel       adaptive layer
─────────────────        ────────────────────────────        ──────────────
Yahoo Finance  ─┐        EMA trend      Supertrend            weights by hit rate
Stooq (backup) ─┼──► ──► MACD           ADX / DI      ──► ──► per-volatility-regime
NSE chain      ─┤        RSI            Bollinger              probability calibration
local cache    ─┘        Donchian       VIX regime             ↑
                         Price action   Seasonality            │ feedback
                         PCR / OI       Max pain               │
                         Llama 3.3, DeepSeek R1, …             │
                                   │                           │
                                   ▼                           │
                         weighted ensemble ──► CALL / PUT ──► outcome scored
                                                    │          next session
                                                    ▼
                                           options trade plan
                                    (strike, premium, SL, target, size)
                                                    │
                                                    ▼
                                      GitHub Pages dashboard
```

Each of the 13 technical strategies is an independent voter scoring the next
session from −1 (strongly bearish → PUT) to +1 (strongly bullish → CALL). A
strategy that cannot see the data it needs **abstains** instead of guessing, and
the ensemble renormalises around it.

### What makes it improve over time

This is the part that matters, and it is all driven by the system's own track record.

**1. Strategy weights follow hit rates.** Every voter carries decayed win/loss
counters. Its hit rate is a Beta posterior and its weight is proportional to its
edge over a coin flip. A strategy that stops working shrinks toward the floor; a
strategy that works takes more of the vote. The decay factor (`learning.decay`)
means recent evidence outweighs old evidence, so the model tracks a changing
market rather than averaging over all history.

**2. Weights are learned per volatility regime.** The same counters are kept
separately for low / normal / high volatility tape, because a breakout strategy
that shines in a trending market is often noise in a quiet one. Small regime
samples are shrunk toward the global estimate, so a regime needs real evidence
before it is allowed to diverge.

**3. Probabilities are calibrated against reality.** The raw ensemble score is
mapped to a probability using the observed frequency of up-moves in each score
bucket, blended with a logistic prior. That is what turns "score +0.42" into
"62% chance of an up move" — and why early predictions are honest about being
uncertain. When the calibrated edge is too thin, the system returns
**NO_TRADE** instead of inventing conviction.

Crucially, each voter is scored on **its own call**, not on the ensemble verdict.
A good strategy still earns credit on a day the ensemble as a whole got it wrong.

### The open-source LLM panel

Open-weight LLMs act as extra voters alongside the technical strategies. Each
configured model becomes its own voter (`llm:<model>`), so the learner tracks
them independently — **an LLM earns weight only if its calls are right**, exactly
like every other strategy. It is never trusted by default.

The models receive only pre-computed indicator numbers, never raw text from the
internet, and their replies are parsed as strict JSON and clamped to range. A
malformed or unreachable model abstains rather than polluting the ensemble.

Any OpenAI-compatible endpoint works. Defaults target free tiers:

| Provider | Models | Cost |
|---|---|---|
| **Groq** (default) | `llama-3.3-70b-versatile`, `openai/gpt-oss-120b` | free tier |
| **OpenRouter** | `meta-llama/llama-3.3-70b-instruct:free`, `deepseek/deepseek-r1:free` | free tier |
| **Together** | `meta-llama/Llama-3.3-70B-Instruct-Turbo-Free` | free tier |
| **Ollama** | `llama3.1`, `qwen2.5` on your own machine | free, offline |

**The panel is optional and currently OFF.** No `LLM_API_KEY` secret is set, so
zero LLMs are running today — the 13 technical strategies carry the whole signal.
The models listed above are what *will* vote once a key is added; nothing calls
out to a model until then, and the system is fully functional for free without
ever doing so.

---

## Setup

### 1. Turn on GitHub Pages (one-time, required)

This is the only manual step, and it must be done by a repository admin — the
Actions token is not permitted to create a Pages site on its own.

1. Go to **Settings → Pages**
   ([direct link](https://github.com/ramanauday1561/kite-mcp/settings/pages))
2. Under **Source**, choose **GitHub Actions**
3. Re-run **Actions → NIFTY 50 F&O Signal → Run workflow**

The dashboard is then served at
**https://ramanauday1561.github.io/kite-mcp/**

Until that is done the analysis still runs and commits its results every trading
day — only the published page is missing, so the Pages deploy step is marked
non-blocking and will not fail the daily run.

If Actions are disabled on a new fork, enable them under **Actions → I understand
my workflows, go ahead and enable them** first.

### 2. Turn on the LLM panel (optional)

Get a free key from [Groq](https://console.groq.com) or
[OpenRouter](https://openrouter.ai), then add it under
**Settings → Secrets and variables → Actions**:

| Type | Name | Value |
|---|---|---|
| Secret | `LLM_API_KEY` | your key |
| Variable | `LLM_PROVIDER` | `groq` (default), `openrouter`, `together`, `huggingface`, `ollama` |
| Variable | `LLM_MODELS` | optional comma-separated override |

Verify it took effect with:

```bash
python -m engine.llm --check     # locally, with LLM_API_KEY exported
```

It prints whether a key is present (never the key itself), polls each model
and shows what each one answered. The workflow runs the same check and reports
it in the log, without ever blocking the signal.

**Treat the key as a secret.** Put it only in GitHub's secret store — never in
`config.json`, a commit, or a chat window. If one is ever exposed, delete it at
the provider and issue a new one; that costs nothing and takes a minute.

### 3. Tune the strategy (optional)

Everything is in [`config.json`](config.json) — lot size, expiry weekday, risk
per trade, the no-trade band, target delta, decay rate and regime boundaries.

> **Check `expiry.weekly_weekday` before trading.** NSE has changed the NIFTY
> weekly expiry day more than once. It defaults to Tuesday (`1`), and the live
> option chain overrides it when reachable, but confirm it against the exchange.

---

## Running locally

```bash
pip install -r requirements.txt

python -m engine.run                 # generate today's signal
python -m engine.run --dry-run       # analyse and print, write nothing
python -m engine.run --no-llm        # technical strategies only

python -m tools.bootstrap --sessions 180   # warm-start on real history
python -m tools.simulate --synthetic 400 --control   # replay the learning loop
python -m unittest discover -s tests -v    # 63 tests

python -m http.server 8000 --directory docs   # view the dashboard
```

### Warm start

A cold model has uniform weights and no calibration, and would need months of
live running to become useful. `tools.bootstrap` replays the last ~180 real
sessions through the full learning loop first, so day one starts with learned
weights and a populated calibration curve. Walk-forward discipline is enforced:
each simulated signal only ever sees candles up to that session, so nothing
leaks backward from the future. The workflow runs it automatically on first use.

---

## Project layout

```
engine/
  datafeed.py     free data sources, fallbacks and the local cache
  indicators.py   EMA, RSI, MACD, ATR, ADX, Supertrend, Bollinger, … (numpy only)
  strategies.py   the 13 independent strategy voters
  llm.py          the open-source LLM panel
  learner.py      adaptive weights, regimes and probability calibration
  evaluate.py     outcome resolution and the performance scoreboard
  options.py      Black-Scholes, expiry maths, strike selection, sizing
  run.py          pipeline entry point
tools/
  bootstrap.py    warm-start the model on real history
  simulate.py     replay the learning loop for testing
docs/             the GitHub Pages dashboard (+ published JSON)
state/            the model's memory, committed so it survives between runs
tests/            63 unit and end-to-end tests
```

The engine depends only on `numpy` and `requests` — no TA-Lib, pandas or scipy,
so a CI install takes seconds. Black-Scholes uses `math.erf`, which is why scipy
is not needed.

### Published data

The dashboard is static and reads these files, which are also a plain JSON API:

| File | Contents |
|---|---|
| `docs/data/latest.json` | current signal, trade plan, every vote and rationale |
| `docs/data/history.json` | recent signals with their resolved outcomes |
| `docs/data/performance.json` | accuracy, win rate, account curve, drawdown |
| `docs/data/model.json` | learned weights, hit rates and the calibration table |
| `docs/data/signals.csv` | **every** signal ever produced, flat CSV for Excel/pandas |

### How often it updates

The page is **generated on a schedule, not on page load** — refreshing your
browser re-reads the last published file, it does not recompute a signal. New
data is published on weekdays at:

| IST | What runs |
|---|---|
| 08:15 | pre-open signal for the session ahead |
| 09:30 – 15:00, every 30 min | intraday refresh while NSE is open |
| 16:15 | post-close signal, resolves the day's prediction |

Actions minutes are unlimited on public repositories, so the intraday cadence
costs nothing. GitHub queues scheduled jobs on shared infrastructure, so a
firing can land a few minutes late; the site always shows the real generation
time and when the next signal is due rather than implying live data.

Intraday runs never score a prediction against the session currently trading —
that candle is still moving, and locking in an outcome against a mid-session
price would corrupt both the track record and the learned weights. A prediction
is resolved only once its forecast session has actually closed.

### Where the data is kept

Nothing is thrown away. Three layers, all committed to the repository:

| Path | Retention |
|---|---|
| `state/archive/YYYY-MM.jsonl` | **append-only, permanent.** One JSON object per line, never rewritten or trimmed. |
| `docs/data/signals.csv` | full flat export of the archive, regenerated each run |
| `state/history.json` | rolling working set the resolver operates on (capped) |
| `docs/data/history.json` | trimmed slice for the site, so visitors download ~50KB not megabytes |

The rolling files are bounded on purpose; the archive and the CSV are the
permanent record. Re-running the pipeline is idempotent — a session already
archived is never written twice. Because it all lives in git, every row also
carries an audit trail of exactly when it appeared.

To analyse the whole history:

```python
import pandas as pd
df = pd.read_csv("https://ramanauday1561.github.io/kite-mcp/data/signals.csv")
print(df.groupby("regime")["bias_correct"].mean())
```

---

## Reading the results honestly

- **Accuracy excludes flat sessions.** A move smaller than
  `signal.move_threshold_pct` is recorded but not scored, because calling
  direction on noise is not a skill.
- **The account curve is position-sized, not a sum of premium returns.** Each
  trade deploys only the premium the risk rule allows, and returns compound on
  that. Summing raw option percentages would imply betting the whole account on
  every signal and would wildly overstate results.
- **Simulated P&L excludes brokerage, slippage and the bid-ask spread.** Real
  results would be lower, and on weekly options the spread is not a rounding error.
- **Premiums are Black-Scholes theoretical values**, not live traded prices. Use
  them as a reference for structure and sizing, not as an entry price.
- **Synthetic-data results are not a forecast.** The numbers from
  `tools.simulate --synthetic` come from data with a deliberately learnable
  pattern. They test the machinery, not the edge. Live accuracy will be far lower.
- **A realistic directional edge is small.** Treat sustained accuracy in the
  low-to-mid 50s as a good outcome, and be suspicious of anything much higher.

## Limitations

- Daily close data only — no intraday signals.
- The NSE option chain blocks most datacenter IPs, so the PCR and max-pain
  voters often abstain on GitHub's runners. The dashboard always shows whether
  they were available.
- No live order placement. This produces analysis, not execution.
- India VIX is used as an IV proxy when the live chain is unavailable.

## Licence

MIT
