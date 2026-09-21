"""Open-source LLMs as additional strategy voters.

Any OpenAI-compatible endpoint works, so the same code drives Groq, OpenRouter,
Together, Hugging Face, Cloudflare Workers AI or a local Ollama. The defaults
point at open-weight models on free tiers:

    Groq        llama-3.3-70b-versatile, openai/gpt-oss-120b   (free tier)
    OpenRouter  meta-llama/llama-3.3-70b-instruct:free,
                deepseek/deepseek-r1:free, qwen/qwen-2.5-72b-instruct:free
    Ollama      llama3.1 / qwen2.5 on your own machine          (free, offline)

Configuration is entirely through environment variables, so the API key lives
in a GitHub Actions secret and never touches the repo:

    LLM_API_KEY    the key (absent -> every LLM voter abstains, run continues)
    LLM_PROVIDER   groq | openrouter | together | huggingface | ollama | custom
    LLM_BASE_URL   override for `custom`
    LLM_MODELS     comma-separated model ids; each becomes its own voter

Each model is a *separate* voter named `llm:<model>`, so the learner tracks
their hit rates independently: a model that calls the market well earns weight,
one that does not decays to the floor. The LLM is fed only the computed
indicator features -- never raw prompts from the internet -- and its reply is
parsed as strict JSON and clamped. A malformed or out-of-range answer abstains
rather than polluting the ensemble.
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

import requests

from .strategies import StrategyVote

PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "default_models": ["llama-3.3-70b-versatile", "openai/gpt-oss-120b"],
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "default_models": [
            "meta-llama/llama-3.3-70b-instruct:free",
            "deepseek/deepseek-r1:free",
        ],
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "default_models": ["meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"],
    },
    "huggingface": {
        "base_url": "https://router.huggingface.co/v1",
        "default_models": ["meta-llama/Llama-3.3-70B-Instruct"],
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "default_models": ["llama3.1", "qwen2.5"],
    },
}

SYSTEM_PROMPT = (
    "You are a quantitative analyst for NIFTY 50 index options. You are given "
    "pre-computed technical indicators for the most recent session. Judge the "
    "direction of the NEXT trading session only.\n"
    "Reply with STRICT JSON and nothing else, in exactly this shape:\n"
    '{"score": <float -1..1>, "confidence": <float 0..1>, "rationale": "<max 40 words>"}\n'
    "score > 0 means bullish (favours a CALL), score < 0 means bearish (favours "
    "a PUT), and 0 means no edge. Be decisive but calibrated: reserve "
    "|score| > 0.7 for genuinely strong setups, and return a score near 0 when "
    "the indicators conflict. Do not invent data that is not provided."
)


def _config() -> Dict:
    provider = os.environ.get("LLM_PROVIDER", "groq").strip().lower()
    preset = PROVIDERS.get(provider, {})
    base_url = os.environ.get("LLM_BASE_URL", "").strip() or preset.get("base_url", "")
    models_env = os.environ.get("LLM_MODELS", "").strip()
    models = ([m.strip() for m in models_env.split(",") if m.strip()]
              if models_env else list(preset.get("default_models", [])))
    return {
        "provider": provider,
        "base_url": base_url.rstrip("/"),
        "models": models[:4],  # cap the fan-out so a run stays inside free limits
        "api_key": os.environ.get("LLM_API_KEY", "").strip(),
        "timeout": int(os.environ.get("LLM_TIMEOUT", "45")),
    }


def is_enabled() -> bool:
    cfg = _config()
    if not cfg["base_url"] or not cfg["models"]:
        return False
    # A local Ollama needs no key; every hosted provider does.
    return bool(cfg["api_key"]) or cfg["provider"] == "ollama"


def voter_names() -> List[str]:
    cfg = _config()
    return [f"llm:{model}" for model in cfg["models"]] if is_enabled() else []


def _round(value, digits=2):
    return round(value, digits) if isinstance(value, (int, float)) else None


def build_market_brief(features: Dict) -> Dict:
    """The compact, numbers-only view of the tape handed to the model."""
    chain = features.get("chain") or {}
    spot = features.get("spot")

    def pct_of_spot(value):
        if value is None or not spot:
            return None
        return round((value - spot) / spot * 100.0, 2)

    return {
        "index": "NIFTY 50",
        "as_of_session": features.get("date"),
        "spot": _round(spot),
        "prev_close": _round(features.get("prev_close")),
        "session_change_pct": _round(
            (spot - features["prev_close"]) / features["prev_close"] * 100.0
            if features.get("prev_close") else None
        ),
        "close_position_in_range": _round(features.get("close_position")),
        "trend": {
            "pct_vs_ema21": pct_of_spot(features.get("ema21")),
            "pct_vs_ema50": pct_of_spot(features.get("ema50")),
            "pct_vs_ema200": pct_of_spot(features.get("ema200")),
            "supertrend": ("bullish" if (features.get("supertrend_dir") or 0) > 0
                           else "bearish" if features.get("supertrend_dir") else None),
            "adx": _round(features.get("adx"), 1),
            "plus_di": _round(features.get("plus_di"), 1),
            "minus_di": _round(features.get("minus_di"), 1),
        },
        "momentum": {
            "rsi14": _round(features.get("rsi"), 1),
            "macd_histogram": _round(features.get("macd_hist")),
            "roc_5d_pct": _round(features.get("roc5")),
            "roc_20d_pct": _round(features.get("roc20")),
        },
        "volatility": {
            "india_vix": _round(features.get("vix")),
            "vix_change_pct": _round(features.get("vix_change_pct")),
            "vix_percentile_1y": _round(features.get("vix_percentile"), 0),
            "atr_pct": _round(features.get("atr_pct")),
            "realized_vol_pct": _round(features.get("realized_vol")),
            "bollinger_percent_b": _round(features.get("percent_b")),
        },
        "range": {
            "donchian20_high": _round(features.get("donchian_upper"), 0),
            "donchian20_low": _round(features.get("donchian_lower"), 0),
            "price_zscore_20d": _round(features.get("zscore20")),
        },
        "option_chain": {
            "pcr_oi": chain.get("pcr_oi"),
            "max_pain": chain.get("max_pain"),
            "pct_to_max_pain": pct_of_spot(chain.get("max_pain")),
            "heaviest_call_oi_strike": chain.get("resistance_strike"),
            "heaviest_put_oi_strike": chain.get("support_strike"),
            "atm_iv": chain.get("atm_iv"),
        } if chain else "unavailable this run",
    }


def _extract_json(text: str) -> Optional[Dict]:
    """Pull the JSON object out of a reply, tolerating fences and reasoning preambles."""
    if not text:
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"```(?:json)?", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for match in re.finditer(r"\{[^{}]*\}", cleaned, re.DOTALL):
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
    return None


def _call_model(cfg: Dict, model: str, brief: Dict) -> Dict:
    headers = {"Content-Type": "application/json"}
    if cfg["api_key"]:
        headers["Authorization"] = f"Bearer {cfg['api_key']}"
    if cfg["provider"] == "openrouter":
        headers["HTTP-Referer"] = "https://github.com/ramanauday1561/kite-mcp"
        headers["X-Title"] = "NIFTY 50 F&O Strategy Builder"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(brief, separators=(",", ":"))},
        ],
        "temperature": 0.2,
        "max_tokens": 400,
    }
    response = requests.post(
        f"{cfg['base_url']}/chat/completions",
        headers=headers, json=payload, timeout=cfg["timeout"],
    )
    if response.status_code != 200:
        raise RuntimeError(f"http {response.status_code}: {response.text[:120]}")
    content = response.json()["choices"][0]["message"]["content"]
    parsed = _extract_json(content)
    if not parsed or "score" not in parsed:
        raise ValueError("model did not return the requested JSON")
    return parsed


def run_llm_votes(features: Dict) -> tuple:
    """Poll each configured open-source model. Returns (votes, notes)."""
    cfg = _config()
    notes: List[str] = []

    if not is_enabled():
        if not cfg["models"] or not cfg["base_url"]:
            notes.append("LLM panel off: no provider/model configured")
        else:
            notes.append("LLM panel off: LLM_API_KEY not set (set it as a repo secret to enable)")
        return [], notes

    brief = build_market_brief(features)
    votes: List[StrategyVote] = []

    for model in cfg["models"]:
        name, label = f"llm:{model}", f"LLM · {model.split('/')[-1]}"
        try:
            parsed = _call_model(cfg, model, brief)
            score = max(-1.0, min(1.0, float(parsed.get("score", 0.0))))
            confidence = max(0.0, min(1.0, float(parsed.get("confidence", abs(score)))))
            rationale = str(parsed.get("rationale", "")).strip()[:300] or "no rationale returned"
            votes.append(StrategyVote(name, label, score, confidence,
                                      f"{rationale} (via {cfg['provider']})"))
        except Exception as exc:
            notes.append(f"{model}: {type(exc).__name__}: {str(exc)[:100]}")
            votes.append(StrategyVote(name, label, 0.0, 0.0,
                                      f"model unreachable this run ({type(exc).__name__})",
                                      abstained=True))
    return votes, notes


# --------------------------------------------------------------------------
# Self-test:  python -m engine.llm --check
# --------------------------------------------------------------------------

def _check() -> int:
    """Verify the configured key and models actually answer, and show what they say.

    Exits non-zero if the panel is off or every model failed, so CI can gate on
    it. Never prints the key -- only whether one is present and its length.
    """
    import sys

    cfg = _config()
    key = cfg["api_key"]
    print("LLM panel configuration")
    print(f"  provider : {cfg['provider']}")
    print(f"  base_url : {cfg['base_url'] or '(none)'}")
    print(f"  api_key  : {'set, ' + str(len(key)) + ' chars' if key else 'NOT SET'}")
    print(f"  models   : {', '.join(cfg['models']) or '(none)'}")

    if not is_enabled():
        print("\nPanel is OFF - the technical strategies will carry the signal alone.")
        if not key and cfg["provider"] != "ollama":
            print("Set LLM_API_KEY (a repository secret in CI) to switch it on.")
        return 1

    # A tiny, representative brief so the check costs almost nothing.
    brief = {
        "index": "NIFTY 50", "as_of_session": "self-test", "spot": 25000,
        "trend": {"pct_vs_ema21": -0.4, "adx": 22.0},
        "momentum": {"rsi14": 44.0, "macd_histogram": -12.0},
        "volatility": {"india_vix": 12.5, "vix_change_pct": -1.2},
    }

    print("\nPolling each model ...")
    ok = 0
    for model in cfg["models"]:
        try:
            parsed = _call_model(cfg, model, brief)
            score = float(parsed.get("score", 0.0))
            print(f"  [ok]   {model}")
            print(f"         score {score:+.2f}  "
                  f"confidence {float(parsed.get('confidence', 0)):.2f}")
            print(f"         \"{str(parsed.get('rationale', ''))[:90]}\"")
            ok += 1
        except Exception as exc:
            print(f"  [FAIL] {model}")
            print(f"         {type(exc).__name__}: {str(exc)[:160]}")

    print(f"\n{ok}/{len(cfg['models'])} model(s) responded.")
    if ok:
        print("Panel is working. These models now vote alongside the 13 technical"
              " strategies and earn weight from their own hit rate.")
        return 0
    print("No model responded - check the key, the provider and the model ids.")
    return 1


if __name__ == "__main__":
    raise SystemExit(_check())
