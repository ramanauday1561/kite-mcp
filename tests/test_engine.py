"""Test suite for the NIFTY F&O strategy engine (stdlib unittest, no extra deps)."""
from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import (archive, datafeed, envfile, evaluate, indicators as ind,
                    learner, llm, options, strategies)
from tools.simulate import replay, synthetic_candles, synthetic_vix
from engine.run import load_config


class TestIndicators(unittest.TestCase):
    def test_sma_matches_hand_calculation(self):
        values = [1, 2, 3, 4, 5, 6]
        result = ind.sma(values, 3)
        self.assertTrue(math.isnan(result[1]))
        self.assertAlmostEqual(result[2], 2.0)
        self.assertAlmostEqual(result[5], 5.0)

    def test_ema_seeds_with_sma_and_tracks_price(self):
        values = list(range(1, 51))
        result = ind.ema(values, 10)
        self.assertAlmostEqual(result[9], sum(range(1, 11)) / 10)
        self.assertLess(result[-1], values[-1])   # lags a rising series
        self.assertGreater(result[-1], values[-10])

    def test_rsi_bounds_and_extremes(self):
        rising = ind.rsi(list(np.arange(1, 60, dtype=float)))
        self.assertAlmostEqual(ind.last(rising), 100.0, places=4)
        falling = ind.rsi(list(np.arange(60, 1, -1, dtype=float)))
        self.assertLess(ind.last(falling), 1.0)

    def test_atr_is_positive_and_tracks_range(self):
        n = 60
        close = np.full(n, 100.0)
        high, low = close + 5, close - 5
        self.assertAlmostEqual(ind.last(ind.atr(high, low, close)), 10.0, places=6)

    def test_adx_flags_a_strong_trend(self):
        n = 120
        close = np.arange(100.0, 100.0 + n)
        adx_values, plus_di, minus_di = ind.adx(close + 1, close - 1, close)
        self.assertGreater(ind.last(adx_values), 40.0)
        self.assertGreater(ind.last(plus_di), ind.last(minus_di))

    def test_last_returns_none_instead_of_nan(self):
        self.assertIsNone(ind.last(np.array([np.nan, np.nan])))
        self.assertIsNone(ind.last(np.array([])))
        self.assertEqual(ind.last(np.array([1.0, 2.0])), 2.0)

    def test_percentile_rank_ranges(self):
        rising = list(np.arange(0, 300, dtype=float))
        self.assertGreater(ind.percentile_rank(rising), 95.0)


class TestOptions(unittest.TestCase):
    def test_put_call_parity(self):
        spot, strike, t, iv, rate = 25000.0, 24800.0, 10 / 365, 0.14, 0.065
        call = options.bs_price(spot, strike, t, iv, "CE", rate)["price"]
        put = options.bs_price(spot, strike, t, iv, "PE", rate)["price"]
        self.assertAlmostEqual(call - put, spot - strike * math.exp(-rate * t), places=1)

    def test_delta_bounds(self):
        call = options.bs_price(25000, 25000, 7 / 365, 0.13, "CE")
        put = options.bs_price(25000, 25000, 7 / 365, 0.13, "PE")
        self.assertTrue(0 < call["delta"] < 1)
        self.assertTrue(-1 < put["delta"] < 0)
        self.assertAlmostEqual(call["delta"] - put["delta"], 1.0, places=3)

    def test_deep_itm_and_otm_behave(self):
        itm = options.bs_price(25000, 20000, 7 / 365, 0.13, "CE")
        otm = options.bs_price(25000, 30000, 7 / 365, 0.13, "CE")
        self.assertGreater(itm["delta"], 0.97)
        self.assertLess(otm["delta"], 0.03)
        self.assertGreater(itm["price"], 4900)

    def test_expired_option_is_worth_intrinsic(self):
        self.assertAlmostEqual(
            options.bs_price(25100, 25000, 0, 0.13, "CE")["price"], 100.0)
        self.assertAlmostEqual(
            options.bs_price(24900, 25000, 0, 0.13, "PE")["price"], 100.0)

    def test_strike_selection_hits_target_delta(self):
        picked = options.pick_strike(25000, "CE", 0.40, 0.13, 7 / 365, 50)
        self.assertLess(abs(abs(picked["delta"]) - 0.40), 0.06)
        self.assertEqual(picked["strike"] % 50, 0)

    def test_expiry_helpers(self):
        from datetime import date
        self.assertEqual(options.next_weekly_expiry(date(2026, 9, 21), 1),
                         date(2026, 9, 22))          # Monday -> Tuesday
        self.assertEqual(options.monthly_expiry(2026, 9, 1), date(2026, 9, 29))
        self.assertEqual(options.parse_nse_expiry("30-Sep-2026").isoformat(), "2026-09-30")
        self.assertIsNone(options.parse_nse_expiry("garbage"))

    def test_expiry_rolls_off_expiry_day(self):
        from datetime import date
        resolved = options.resolve_expiry(date(2026, 9, 22), 1, None)  # a Tuesday
        self.assertEqual(resolved["date"], "2026-09-29")

    def test_live_chain_expiry_wins_over_calculation(self):
        from datetime import date
        resolved = options.resolve_expiry(date(2026, 9, 21), 1, {"expiry": "24-Sep-2026"})
        self.assertEqual(resolved["source"], "nse_option_chain")
        self.assertEqual(resolved["date"], "2026-09-24")

    def test_trade_plan_is_internally_consistent(self):
        risk = load_config()["risk"]
        expiry = {"date": "2026-09-29", "days_to_expiry": 8, "source": "test"}
        plan = options.build_trade(25000, "CALL", 13.0, expiry, risk, 75, 50, 30)
        self.assertEqual(plan["option_type"], "CE")
        self.assertEqual(plan["structure"], "long_option")
        self.assertLess(plan["stop_loss"], plan["theoretical_premium"])
        self.assertGreater(plan["target"], plan["theoretical_premium"])
        self.assertGreaterEqual(plan["suggested_lots"], 0)
        self.assertLessEqual(plan["capital_at_risk"],
                             risk["capital"] * risk["risk_per_trade_pct"] / 100.0 + 1)
        self.assertGreater(plan["breakeven"], plan["strike"])

    def test_high_iv_switches_to_a_spread(self):
        risk = load_config()["risk"]
        expiry = {"date": "2026-09-29", "days_to_expiry": 8, "source": "test"}
        plan = options.build_trade(25000, "PUT", 24.0, expiry, risk, 75, 50, 90)
        self.assertEqual(plan["structure"], "debit_spread")
        self.assertIn("short_leg", plan)
        self.assertLess(plan["net_debit"], plan["theoretical_premium"])
        self.assertGreater(plan["max_profit"], 0)

    def test_simulated_pnl_directionality(self):
        risk = load_config()["risk"]
        expiry = {"date": "2026-09-29", "days_to_expiry": 8, "source": "test"}
        plan = options.build_trade(25000, "CALL", 13.0, expiry, risk, 75, 50, 30)
        up = options.simulate_pnl(plan, 25000, 25400, 13.0, 1)
        down = options.simulate_pnl(plan, 25000, 24600, 13.0, 1)
        self.assertGreater(up["pnl_pct"], 0)
        self.assertLess(down["pnl_pct"], 0)
        self.assertGreaterEqual(down["pnl_pct"], -100.0)


class TestLearner(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()["learning"]
        self.names = ["a", "b", "c"]

    def test_cold_start_weights_are_equal(self):
        weights = learner.strategy_weights(learner.new_state(), "mid_vol", self.names, self.cfg)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)
        self.assertAlmostEqual(min(weights.values()), max(weights.values()), places=9)

    def test_a_winning_strategy_earns_weight(self):
        state = learner.new_state()
        learn_cfg = {**self.cfg, "flat_band": 0.0}
        for _ in range(40):
            record = {"regime": "mid_vol", "score": 0.5, "votes": {
                "a": {"score": 0.8, "abstained": False},     # always right
                "b": {"score": -0.8, "abstained": False},    # always wrong
                "c": {"score": 0.0, "abstained": True},      # abstains
            }}
            learner.apply_outcome(state, record, +1.0, 20.0, learn_cfg)
        weights = learner.strategy_weights(state, "mid_vol", self.names, self.cfg)
        self.assertGreater(weights["a"], weights["b"])
        self.assertGreater(weights["a"], weights["c"])
        self.assertGreater(weights["a"], 0.6)
        stats = learner.strategy_stats(state, "mid_vol", self.names, self.cfg)
        self.assertGreater(stats["a"]["hit_rate"], 0.8)
        self.assertLess(stats["b"]["hit_rate"], 0.2)
        self.assertEqual(stats["c"]["sample"], 0)

    def test_weights_respect_the_configured_cap(self):
        state = learner.new_state()
        learn_cfg = {**self.cfg, "flat_band": 0.0}
        for _ in range(60):
            learner.apply_outcome(state, {"regime": "mid_vol", "score": 0.5, "votes": {
                "a": {"score": 1.0, "abstained": False}}}, +1.0, 10.0, learn_cfg)
        weights = learner.strategy_weights(state, "mid_vol", self.names, self.cfg)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)
        # 'a' dominates but the floor keeps the others alive for re-evaluation.
        self.assertGreater(min(weights.values()), 0.0)

    def test_decay_lets_the_model_change_its_mind(self):
        state = learner.new_state()
        learn_cfg = {**self.cfg, "flat_band": 0.0}
        for _ in range(30):   # 'a' is right for a while ...
            learner.apply_outcome(state, {"regime": "mid_vol", "score": 0.5, "votes": {
                "a": {"score": 0.9, "abstained": False}}}, +1.0, 5.0, learn_cfg)
        early = learner.strategy_stats(state, "mid_vol", ["a"], self.cfg)["a"]["hit_rate"]
        for _ in range(60):   # ... then it stops working
            learner.apply_outcome(state, {"regime": "mid_vol", "score": 0.5, "votes": {
                "a": {"score": 0.9, "abstained": False}}}, -1.0, -5.0, learn_cfg)
        late = learner.strategy_stats(state, "mid_vol", ["a"], self.cfg)["a"]["hit_rate"]
        self.assertGreater(early, 0.8)
        self.assertLess(late, 0.2)

    def test_regimes_are_learned_separately(self):
        state = learner.new_state()
        learn_cfg = {**self.cfg, "flat_band": 0.0}
        for _ in range(40):
            learner.apply_outcome(state, {"regime": "high_vol", "score": 0.5, "votes": {
                "a": {"score": 0.9, "abstained": False}}}, +1.0, 5.0, learn_cfg)
            learner.apply_outcome(state, {"regime": "low_vol", "score": 0.5, "votes": {
                "a": {"score": 0.9, "abstained": False}}}, -1.0, -5.0, learn_cfg)
        high = learner.strategy_weights(state, "high_vol", self.names, self.cfg)["a"]
        low = learner.strategy_weights(state, "low_vol", self.names, self.cfg)["a"]
        self.assertGreater(high, low)

    def test_abstaining_strategies_are_never_scored(self):
        state = learner.new_state()
        learner.apply_outcome(state, {"regime": "mid_vol", "score": 0.4, "votes": {
            "c": {"score": 0.0, "abstained": True}}}, +1.0, None,
            {**self.cfg, "flat_band": 0.0})
        self.assertNotIn("c", state["strategies"])

    def test_calibration_starts_from_the_logistic_prior(self):
        state = learner.new_state()
        bullish = learner.calibrate(state, 0.8)
        bearish = learner.calibrate(state, -0.8)
        neutral = learner.calibrate(state, 0.0)
        self.assertGreater(bullish["probability_up"], 0.6)
        self.assertLess(bearish["probability_up"], 0.4)
        self.assertAlmostEqual(neutral["probability_up"], 0.5, places=6)
        self.assertEqual(neutral["sample"], 0)

    def test_calibration_learns_from_observed_frequencies(self):
        state = learner.new_state()
        learn_cfg = {**self.cfg, "flat_band": 0.0}
        for _ in range(60):   # score +0.5 keeps resolving DOWN
            learner.apply_outcome(state, {"regime": "mid_vol", "score": 0.5, "votes": {
                "a": {"score": 0.5, "abstained": False}}}, -1.0, None, learn_cfg)
        calibrated = learner.calibrate(state, 0.5)
        self.assertLess(calibrated["probability_up"], 0.5)
        self.assertGreater(calibrated["sample"], 0)
        self.assertIn("resolved signals", calibrated["basis"])

    def test_decide_stands_aside_inside_the_band(self):
        cfg = load_config()["signal"]
        calm = learner.calibrate(learner.new_state(), 0.02)
        decision = learner.decide(0.02, calm, cfg)
        self.assertEqual(decision["action"], "NO_TRADE")
        self.assertIn("no-trade band", decision["reason"])

    def test_decide_issues_a_call_and_a_put(self):
        cfg = load_config()["signal"]
        state = learner.new_state()
        bull = learner.decide(0.75, learner.calibrate(state, 0.75), cfg)
        bear = learner.decide(-0.75, learner.calibrate(state, -0.75), cfg)
        self.assertEqual(bull["action"], "CALL")
        self.assertEqual(bear["action"], "PUT")
        self.assertGreater(bull["confidence"], 0.55)

    def test_regime_classification(self):
        bounds = self.cfg["vix_regime_bounds"]
        self.assertEqual(learner.regime_of({"vix": 10.0}, bounds), "low_vol")
        self.assertEqual(learner.regime_of({"vix": 16.0}, bounds), "mid_vol")
        self.assertEqual(learner.regime_of({"vix": 28.0}, bounds), "high_vol")
        # Falls back to realised vol when VIX is missing.
        self.assertEqual(
            learner.regime_of({"vix": None, "realized_vol": 30.0}, bounds), "high_vol")

    def test_combine_renormalises_around_abstentions(self):
        votes = [
            strategies.StrategyVote("a", "A", 1.0, 1.0, ""),
            strategies.StrategyVote("b", "B", 1.0, 1.0, ""),
            strategies.StrategyVote("c", "C", -1.0, 1.0, "", abstained=True),
        ]
        result = learner.combine(votes, {"a": 0.4, "b": 0.4, "c": 0.2})
        self.assertAlmostEqual(result["score"], 1.0, places=6)
        self.assertEqual(result["active"], 2)
        self.assertAlmostEqual(result["agreement"], 1.0)

    def test_combine_handles_everyone_abstaining(self):
        votes = [strategies.StrategyVote("a", "A", 0.0, 0.0, "", abstained=True)]
        result = learner.combine(votes, {"a": 1.0})
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["active"], 0)

    def test_state_round_trips_through_disk(self):
        import tempfile
        state = learner.new_state()
        state["strategies"]["a"] = {"wins": 3.0, "losses": 1.0, "pnl": 5.0, "n": 4.0}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.json")
            learner.save_state(path, state)
            reloaded = learner.load_state(path)
        self.assertEqual(reloaded["strategies"]["a"]["wins"], 3.0)

    def test_corrupt_state_file_does_not_crash(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.json")
            with open(path, "w") as handle:
                handle.write("{not json")
            self.assertEqual(learner.load_state(path)["runs"], 0)


class TestStrategies(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.candles = synthetic_candles(300, seed=3)
        cls.vix = synthetic_vix(cls.candles, seed=3)

    def test_every_strategy_returns_a_bounded_score(self):
        features = strategies.build_features(self.candles, self.vix, None)
        votes = strategies.run_all(features)
        self.assertEqual(len(votes), len(strategies.STRATEGIES))
        for vote in votes:
            self.assertTrue(-1.0 <= vote.score <= 1.0, f"{vote.name} out of range")
            self.assertTrue(0.0 <= vote.confidence <= 1.0)
            self.assertTrue(vote.rationale)

    def test_oi_strategies_abstain_without_a_chain(self):
        features = strategies.build_features(self.candles, self.vix, None)
        votes = {v.name: v for v in strategies.run_all(features)}
        self.assertTrue(votes["oi_pcr"].abstained)
        self.assertTrue(votes["max_pain"].abstained)

    def test_oi_strategies_engage_with_a_chain(self):
        spot = self.candles.close[-1]
        chain = {"pcr_oi": 1.35, "max_pain": spot + 150, "call_oi_change": 1e5,
                 "put_oi_change": 5e5, "support_strike": spot - 200,
                 "resistance_strike": spot + 300, "atm_iv": 12.0}
        features = strategies.build_features(self.candles, self.vix, chain)
        votes = {v.name: v for v in strategies.run_all(features)}
        self.assertFalse(votes["oi_pcr"].abstained)
        self.assertGreater(votes["oi_pcr"].score, 0)      # PCR > 1 with put writing
        self.assertGreater(votes["max_pain"].score, 0)    # max pain above spot

    def test_strategies_abstain_on_short_history(self):
        short = datafeed.Candles(
            dates=self.candles.dates[:12], open=self.candles.open[:12],
            high=self.candles.high[:12], low=self.candles.low[:12],
            close=self.candles.close[:12], volume=self.candles.volume[:12])
        votes = strategies.run_all(strategies.build_features(short, None, None))
        self.assertGreater(sum(1 for v in votes if v.abstained), 4)
        for vote in votes:
            self.assertTrue(-1.0 <= vote.score <= 1.0)

    def test_a_strong_uptrend_reads_bullish_overall(self):
        n = 260
        close = np.linspace(20000, 26000, n) + np.random.default_rng(1).normal(0, 25, n)
        rising = datafeed.Candles(
            dates=self.candles.dates[:n], open=list(close - 10),
            high=list(close + 40), low=list(close - 40), close=list(close),
            volume=[2e6] * n)
        features = strategies.build_features(rising, None, None)
        votes = strategies.run_all(features)
        weights = learner.strategy_weights(
            learner.new_state(), "mid_vol", [v.name for v in votes],
            load_config()["learning"])
        self.assertGreater(learner.combine(votes, weights)["score"], 0.25)

    def test_a_strong_downtrend_reads_bearish_overall(self):
        n = 260
        close = np.linspace(26000, 20000, n) + np.random.default_rng(2).normal(0, 25, n)
        falling = datafeed.Candles(
            dates=self.candles.dates[:n], open=list(close + 10),
            high=list(close + 40), low=list(close - 40), close=list(close),
            volume=[2e6] * n)
        features = strategies.build_features(falling, None, None)
        votes = strategies.run_all(features)
        weights = learner.strategy_weights(
            learner.new_state(), "mid_vol", [v.name for v in votes],
            load_config()["learning"])
        self.assertLess(learner.combine(votes, weights)["score"], -0.25)

    def test_a_broken_strategy_abstains_instead_of_crashing(self):
        def exploding(_features):
            raise RuntimeError("boom")
        original = list(strategies.STRATEGIES)
        strategies.STRATEGIES.append(exploding)
        try:
            votes = strategies.run_all(
                strategies.build_features(self.candles, self.vix, None))
            self.assertTrue(votes[-1].abstained)
            self.assertIn("strategy error", votes[-1].rationale)
        finally:
            strategies.STRATEGIES[:] = original


class TestDataFeed(unittest.TestCase):
    def test_merge_prefers_fresh_data_and_keeps_order(self):
        old = datafeed.Candles(dates=["2026-01-01", "2026-01-02"], open=[1, 2],
                               high=[2, 3], low=[0, 1], close=[1.5, 2.5],
                               volume=[10, 20], source="cache")
        new = datafeed.Candles(dates=["2026-01-02", "2026-01-03"], open=[9, 3],
                               high=[9, 4], low=[9, 2], close=[9.0, 3.5],
                               volume=[99, 30], source="yahoo")
        merged = old.merge(new)
        self.assertEqual(merged.dates, ["2026-01-01", "2026-01-02", "2026-01-03"])
        self.assertEqual(merged.close[1], 9.0)          # fresh wins
        self.assertEqual(merged.source, "yahoo")

    def test_tail_and_round_trip(self):
        candles = synthetic_candles(30, seed=5)
        self.assertEqual(len(candles.tail(10)), 10)
        self.assertEqual(len(candles.tail(0)), 30)
        restored = datafeed.Candles.from_dict(candles.to_dict())
        self.assertEqual(restored.close, candles.close)

    def test_max_pain_finds_the_minimum_payout_strike(self):
        chain = {24000: {"call_oi": 500, "put_oi": 10},
                 24500: {"call_oi": 50, "put_oi": 50},
                 25000: {"call_oi": 10, "put_oi": 500}}
        self.assertEqual(datafeed._max_pain(chain), 24500)

    def test_option_chain_summary(self):
        payload = {"records": {
            "expiryDates": ["30-Sep-2026"], "underlyingValue": 25000,
            "data": [
                {"strikePrice": 24950, "expiryDate": "30-Sep-2026",
                 "CE": {"openInterest": 100, "changeinOpenInterest": 10,
                        "totalTradedVolume": 50, "impliedVolatility": 12.0},
                 "PE": {"openInterest": 200, "changeinOpenInterest": 40,
                        "totalTradedVolume": 90, "impliedVolatility": 13.0}},
                {"strikePrice": 25050, "expiryDate": "30-Sep-2026",
                 "CE": {"openInterest": 300, "changeinOpenInterest": 20,
                        "totalTradedVolume": 70, "impliedVolatility": 12.5},
                 "PE": {"openInterest": 150, "changeinOpenInterest": 5,
                        "totalTradedVolume": 30, "impliedVolatility": 13.5}},
            ]}}
        summary = datafeed.summarise_option_chain(payload, 25000, 50)
        self.assertAlmostEqual(summary["pcr_oi"], 350 / 400, places=4)
        self.assertEqual(summary["atm_strike"], 25000)
        self.assertEqual(summary["resistance_strike"], 25050)
        self.assertEqual(summary["support_strike"], 24950)
        self.assertEqual(summary["expiry"], "30-Sep-2026")

    def test_malformed_chain_returns_none(self):
        self.assertIsNone(datafeed.summarise_option_chain({"nonsense": 1}, 25000))
        self.assertIsNone(datafeed.summarise_option_chain(
            {"records": {"data": [], "expiryDates": []}}, 25000))


class TestLLMPanel(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in
                       ("LLM_API_KEY", "LLM_PROVIDER", "LLM_MODELS", "LLM_BASE_URL")}
        for key in self._saved:
            os.environ.pop(key, None)

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_disabled_without_a_key(self):
        self.assertFalse(llm.is_enabled())
        votes, notes = llm.run_llm_votes({"spot": 25000})
        self.assertEqual(votes, [])
        self.assertTrue(any("LLM_API_KEY" in note for note in notes))

    def test_enabled_with_a_key_and_names_each_model(self):
        os.environ["LLM_API_KEY"] = "test-key"
        os.environ["LLM_PROVIDER"] = "groq"
        self.assertTrue(llm.is_enabled())
        names = llm.voter_names()
        self.assertTrue(all(name.startswith("llm:") for name in names))
        self.assertIn("llm:llama-3.3-70b-versatile", names)

    def test_ollama_needs_no_key(self):
        os.environ["LLM_PROVIDER"] = "ollama"
        self.assertTrue(llm.is_enabled())

    def test_custom_model_list_is_respected_and_capped(self):
        os.environ["LLM_API_KEY"] = "k"
        os.environ["LLM_MODELS"] = "a,b,c,d,e,f"
        self.assertEqual(len(llm.voter_names()), 4)

    def test_json_extraction_tolerates_real_model_output(self):
        cases = {
            '{"score":0.5,"confidence":0.7,"rationale":"x"}': 0.5,
            '```json\n{"score":-0.3,"confidence":0.6,"rationale":"y"}\n```': -0.3,
            '<think>long reasoning</think>{"score":0.9,"confidence":0.8,"rationale":"z"}': 0.9,
            'Sure: {"score":0.1,"confidence":0.2,"rationale":"w"} hope this helps': 0.1,
        }
        for text, expected in cases.items():
            self.assertEqual(llm._extract_json(text)["score"], expected)
        self.assertIsNone(llm._extract_json("no json here"))
        self.assertIsNone(llm._extract_json(""))

    def test_self_check_reports_off_without_a_key(self):
        import contextlib
        import io as _io
        buf = _io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = llm._check()
        self.assertEqual(code, 1)
        self.assertIn("NOT SET", buf.getvalue())
        self.assertIn("Panel is OFF", buf.getvalue())

    def test_self_check_never_prints_the_key(self):
        import contextlib
        import io as _io
        os.environ["LLM_API_KEY"] = "gsk_averysecretvalue"
        os.environ["LLM_PROVIDER"] = "custom"
        os.environ["LLM_BASE_URL"] = "http://127.0.0.1:9"
        os.environ["LLM_MODELS"] = "m"
        os.environ["LLM_TIMEOUT"] = "2"
        buf = _io.StringIO()
        with contextlib.redirect_stdout(buf):
            llm._check()
        self.assertNotIn("averysecretvalue", buf.getvalue())
        self.assertIn("set, ", buf.getvalue())   # reports presence, not value

    def test_market_brief_contains_no_secrets_and_only_numbers(self):
        candles = synthetic_candles(260, seed=11)
        features = strategies.build_features(candles, synthetic_vix(candles), None)
        brief = llm.build_market_brief(features)
        self.assertEqual(brief["index"], "NIFTY 50")
        self.assertIn("momentum", brief)
        self.assertEqual(brief["option_chain"], "unavailable this run")
        serialised = __import__("json").dumps(brief)
        self.assertNotIn("api_key", serialised.lower())

    def test_unreachable_model_abstains_rather_than_voting(self):
        os.environ["LLM_API_KEY"] = "bad-key"
        os.environ["LLM_BASE_URL"] = "http://127.0.0.1:9"   # nothing listening
        os.environ["LLM_PROVIDER"] = "custom"
        os.environ["LLM_MODELS"] = "some-model"
        os.environ["LLM_TIMEOUT"] = "2"
        votes, notes = llm.run_llm_votes({"spot": 25000, "date": "2026-09-21"})
        self.assertEqual(len(votes), 1)
        self.assertTrue(votes[0].abstained)
        self.assertEqual(votes[0].score, 0.0)
        self.assertTrue(notes)


class TestSchedule(unittest.TestCase):
    """The workflow cron and the site's 'next update' must never drift apart."""

    def _workflow_crons(self):
        import re
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, ".github", "workflows", "signal.yml")
        with open(path, encoding="utf-8") as handle:
            return re.findall(r"cron:\s*'([^']+)'", handle.read())

    def _expand(self, cron):
        """Expand the limited cron shapes this workflow uses into (h, m) slots."""
        minute, hour = cron.split()[0], cron.split()[1]
        minutes = [int(x) for x in minute.split(",")]
        if "-" in hour:
            lo, hi = hour.split("-")
            hours = list(range(int(lo), int(hi) + 1))
        else:
            hours = [int(hour)]
        return {(h, m) for h in hours for m in minutes}

    def test_code_schedule_matches_the_workflow_cron(self):
        from engine.run import SCHEDULE_UTC
        from_cron = set()
        for cron in self._workflow_crons():
            from_cron |= self._expand(cron)
        self.assertEqual(from_cron, set(SCHEDULE_UTC),
                         "engine/run.py SCHEDULE_UTC is out of step with signal.yml")

    def test_intraday_runs_avoid_the_congested_minutes(self):
        """:00 and :30 are where every cron on GitHub piles in and gets dropped."""
        from engine.run import SCHEDULE_UTC
        intraday = [(h, m) for h, m in SCHEDULE_UTC if 4 <= h <= 9]
        self.assertTrue(intraday)
        for hour, minute in intraday:
            self.assertNotIn(minute, (0, 30),
                             f"{hour:02d}:{minute:02d} UTC sits in the congested slot")

    def test_ist_slots_cover_the_trading_session(self):
        from engine.run import SCHEDULE_IST
        slots = set(SCHEDULE_IST)
        self.assertIn((8, 15), slots)     # pre-open
        self.assertIn((16, 15), slots)    # post-close
        # NSE trades 09:15-15:30 IST; we should refresh several times inside it.
        during = [(h, m) for h, m in slots if (9, 15) <= (h, m) <= (15, 30)]
        self.assertGreaterEqual(len(during), 8)


class TestEnvFile(unittest.TestCase):
    """A local .env must be convenient without ever risking a leaked key."""

    def test_parses_the_usual_shapes(self):
        parsed = envfile.parse(
            "# a comment\n"
            "\n"
            "LLM_API_KEY=gsk_plain\n"
            "QUOTED=\"gsk_quoted\"\n"
            "SINGLE='gsk_single'\n"
            "export EXPORTED=gsk_exported\n"
            "  SPACED = gsk_spaced  \n"
            "NOT_A_PAIR\n"
        )
        self.assertEqual(parsed["LLM_API_KEY"], "gsk_plain")
        self.assertEqual(parsed["QUOTED"], "gsk_quoted")
        self.assertEqual(parsed["SINGLE"], "gsk_single")
        self.assertEqual(parsed["EXPORTED"], "gsk_exported")
        self.assertEqual(parsed["SPACED"], "gsk_spaced")
        self.assertNotIn("NOT_A_PAIR", parsed)

    def test_a_real_environment_variable_wins(self):
        import tempfile
        os.environ["ENVFILE_TEST_KEY"] = "from-environment"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, ".env")
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write("ENVFILE_TEST_KEY=from-file\n")
                envfile.load(path)
                # CI passes the key as a real secret; a stray file must not win.
                self.assertEqual(os.environ["ENVFILE_TEST_KEY"], "from-environment")
                envfile.load(path, override=True)
                self.assertEqual(os.environ["ENVFILE_TEST_KEY"], "from-file")
        finally:
            os.environ.pop("ENVFILE_TEST_KEY", None)

    def test_missing_or_unreadable_file_is_a_no_op(self):
        self.assertEqual(envfile.load("/nonexistent/path/.env"), {})

    def test_describe_never_reveals_a_secret_value(self):
        summary = envfile.describe({
            "LLM_API_KEY": "gsk_averysecretvalue",
            "SOME_TOKEN": "tok_secret",
            "API_PASSWORD": "hunter2",
            "LLM_PROVIDER": "groq",
        })
        for secret in ("gsk_averysecretvalue", "tok_secret", "hunter2"):
            self.assertNotIn(secret, summary)
        self.assertIn("<hidden", summary)
        self.assertIn("LLM_PROVIDER=groq", summary)   # non-secrets stay readable
        self.assertIsNone(envfile.describe({}))

    def test_dotenv_is_gitignored_but_the_example_is_not(self):
        """Guards the whole point: a key on disk must never be committable."""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, ".gitignore"), encoding="utf-8") as handle:
            ignored = [ln.strip() for ln in handle]
        self.assertIn(".env", ignored)
        self.assertIn("!.env.example", ignored)
        self.assertTrue(os.path.isfile(os.path.join(root, ".env.example")))

    def test_the_example_file_holds_no_real_key(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, ".env.example"), encoding="utf-8") as handle:
            content = handle.read()
        key = envfile.parse(content).get("LLM_API_KEY", "")
        self.assertIn("replace_me", key)
        self.assertLess(len(key), 30)   # a real Groq key is ~56 chars


class TestEvaluate(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        self.candles = datafeed.Candles(
            dates=["2026-09-14", "2026-09-15", "2026-09-16"],
            open=[100, 101, 103], high=[102, 104, 106], low=[99, 100, 102],
            close=[100.0, 102.0, 101.0], volume=[1e6] * 3)

    def _record(self, session, bias, action="CALL", vote_score=None):
        # By default the voter agrees with the headline bias. Voters are scored
        # on their OWN call, not on the ensemble verdict, so the two can diverge.
        if vote_score is None:
            vote_score = 0.5 if bias == "CALL" else -0.5
        return {"id": session, "session_date": session, "spot": 100.0,
                "regime": "mid_vol", "score": 0.4, "probability_up": 0.6,
                "action": action, "bias": bias, "confidence": 0.6,
                "trade": None, "votes": {"a": {"score": vote_score, "abstained": False}},
                "outcome": None}

    def test_correct_call_is_scored_as_a_hit(self):
        history = [self._record("2026-09-14", "CALL")]
        state = learner.new_state()
        count, _ = evaluate.resolve_pending(history, self.candles, None, state, self.cfg)
        self.assertEqual(count, 1)
        outcome = history[0]["outcome"]
        self.assertTrue(outcome["bias_correct"])
        self.assertEqual(outcome["forecast_date"], "2026-09-15")
        self.assertAlmostEqual(outcome["move_pct"], 2.0, places=3)
        self.assertGreater(state["strategies"]["a"]["wins"], 0)

    def test_wrong_put_is_scored_as_a_miss(self):
        history = [self._record("2026-09-14", "PUT", "PUT")]
        state = learner.new_state()
        evaluate.resolve_pending(history, self.candles, None, state, self.cfg)
        self.assertFalse(history[0]["outcome"]["bias_correct"])
        self.assertGreater(state["strategies"]["a"]["losses"], 0)

    def test_voters_are_scored_on_their_own_call_not_the_verdict(self):
        # The ensemble said PUT and was wrong, but this voter said +0.5 (up) and
        # was right. It must be credited a win, otherwise a good strategy could
        # never earn weight back inside a losing ensemble.
        history = [self._record("2026-09-14", "PUT", "PUT", vote_score=0.5)]
        state = learner.new_state()
        evaluate.resolve_pending(history, self.candles, None, state, self.cfg)
        self.assertFalse(history[0]["outcome"]["bias_correct"])
        self.assertGreater(state["strategies"]["a"]["wins"], 0)
        self.assertEqual(state["strategies"]["a"]["losses"], 0)

    def test_the_newest_record_stays_pending(self):
        history = [self._record("2026-09-16", "CALL")]
        count, _ = evaluate.resolve_pending(
            history, self.candles, None, learner.new_state(), self.cfg)
        self.assertEqual(count, 0)
        self.assertIsNone(history[0]["outcome"])

    def test_already_resolved_records_are_not_double_counted(self):
        history = [self._record("2026-09-14", "CALL")]
        state = learner.new_state()
        evaluate.resolve_pending(history, self.candles, None, state, self.cfg)
        count, _ = evaluate.resolve_pending(history, self.candles, None, state, self.cfg)
        self.assertEqual(count, 0)
        self.assertEqual(state["resolved"], 1)

    def test_a_flat_session_is_excluded_from_accuracy(self):
        flat = datafeed.Candles(
            dates=["2026-09-14", "2026-09-15"], open=[100, 100], high=[101, 101],
            low=[99, 99], close=[100.0, 100.02], volume=[1e6, 1e6])
        history = [self._record("2026-09-14", "CALL")]
        evaluate.resolve_pending(history, flat, None, learner.new_state(), self.cfg)
        self.assertTrue(history[0]["outcome"]["flat"])
        self.assertIsNone(history[0]["outcome"]["bias_correct"])
        self.assertIsNone(evaluate.performance(history)["accuracy_all"])

    def test_performance_on_an_empty_history(self):
        perf = evaluate.performance([])
        self.assertEqual(perf["total_signals"], 0)
        self.assertIsNone(perf["accuracy_all"])
        self.assertEqual(perf["equity_curve"], [])

    def test_no_trade_records_score_bias_but_not_pnl(self):
        history = [self._record("2026-09-14", "CALL", action="NO_TRADE")]
        evaluate.resolve_pending(history, self.candles, None, learner.new_state(), self.cfg)
        perf = evaluate.performance(history)
        self.assertEqual(perf["scored"], 1)
        self.assertEqual(perf["trades_simulated"], 0)
        self.assertFalse(history[0]["outcome"]["traded"])

    def test_account_curve_scales_by_capital_deployed(self):
        record = self._record("2026-09-14", "CALL")
        record["trade"] = {"strike": 100, "option_type": "CE", "days_to_expiry": 5,
                           "theoretical_premium": 10.0, "lot_size": 75,
                           "iv_used_pct": 13.0, "capital_deployed_pct": 4.0}
        evaluate.resolve_pending([record], self.candles, None, learner.new_state(), self.cfg)
        perf = evaluate.performance([record])
        point = perf["equity_curve"][0]
        # Account impact must be the premium return scaled down by the sizing rule.
        self.assertLess(abs(point["account_return_pct"]), abs(point["pnl_pct"]))
        self.assertAlmostEqual(point["account_return_pct"],
                               point["pnl_pct"] * 0.04, places=3)


class TestIntradaySafety(unittest.TestCase):
    """Running during market hours must not score against a live candle."""

    def setUp(self):
        self.cfg = load_config()
        # 09-16 is "today" and still trading: its close is not final.
        self.candles = datafeed.Candles(
            dates=["2026-09-14", "2026-09-15", "2026-09-16"],
            open=[100, 101, 103], high=[102, 104, 106], low=[99, 100, 102],
            close=[100.0, 102.0, 101.0], volume=[1e6] * 3)

    def _record(self, session):
        return {"id": session, "session_date": session, "spot": 100.0,
                "regime": "mid_vol", "score": 0.4, "probability_up": 0.6,
                "action": "CALL", "bias": "CALL", "confidence": 0.6,
                "trade": None, "votes": {"a": {"score": 0.5, "abstained": False}},
                "outcome": None}

    def test_does_not_resolve_against_the_current_session(self):
        history = [self._record("2026-09-15")]   # would resolve against 09-16
        count, _ = evaluate.resolve_pending(
            history, self.candles, None, learner.new_state(), self.cfg,
            today="2026-09-16")
        self.assertEqual(count, 0)
        self.assertIsNone(history[0]["outcome"])

    def test_resolves_once_that_session_has_closed(self):
        history = [self._record("2026-09-15")]
        count, _ = evaluate.resolve_pending(
            history, self.candles, None, learner.new_state(), self.cfg,
            today="2026-09-17")     # 09-16 is now a finished session
        self.assertEqual(count, 1)
        self.assertEqual(history[0]["outcome"]["forecast_date"], "2026-09-16")

    def test_older_records_still_resolve_intraday(self):
        # 09-14 -> 09-15 is fully in the past even while 09-16 is trading.
        history = [self._record("2026-09-14")]
        count, _ = evaluate.resolve_pending(
            history, self.candles, None, learner.new_state(), self.cfg,
            today="2026-09-16")
        self.assertEqual(count, 1)
        self.assertEqual(history[0]["outcome"]["forecast_date"], "2026-09-15")

    def test_omitting_today_keeps_replay_behaviour(self):
        history = [self._record("2026-09-15")]
        count, _ = evaluate.resolve_pending(
            history, self.candles, None, learner.new_state(), self.cfg)
        self.assertEqual(count, 1)


class TestArchive(unittest.TestCase):
    """The permanent record must never lose or duplicate a signal."""

    def _record(self, session, resolved=True):
        record = {"session_date": session, "generated_at": f"{session}T16:15:00+05:30",
                  "spot": 25000.0, "regime": "mid_vol", "action": "CALL",
                  "bias": "CALL", "score": 0.4, "probability_up": 0.61,
                  "confidence": 0.61,
                  "trade": {"structure": "long_option", "option_type": "CE",
                            "strike": 25050.0, "expiry": "2026-09-29",
                            "days_to_expiry": 5, "theoretical_premium": 80.0,
                            "stop_loss": 52.0, "target": 132.0},
                  "outcome": None}
        if resolved:
            record["outcome"] = {
                "forecast_date": session, "entry_close": 25000.0,
                "exit_close": 25100.0, "move_pct": 0.4, "flat": False,
                "bias_correct": True, "traded": True,
                "option_pnl_pct": 22.5, "option_pnl_per_lot": 1350.0}
        return record

    def test_appends_and_never_duplicates_on_rerun(self):
        import tempfile
        history = [self._record("2026-09-14"), self._record("2026-09-15")]
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(archive.archive_resolved(history, tmp), 2)
            # A re-run on the same day must not write the rows again.
            self.assertEqual(archive.archive_resolved(history, tmp), 0)
            history.append(self._record("2026-09-16"))
            self.assertEqual(archive.archive_resolved(history, tmp), 1)
            self.assertEqual(len(archive.load_archive(tmp)), 3)

    def test_pending_records_are_not_archived(self):
        import tempfile
        history = [self._record("2026-09-14", resolved=False)]
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(archive.archive_resolved(history, tmp), 0)
            self.assertEqual(archive.load_archive(tmp), [])

    def test_splits_by_month_and_reloads_in_order(self):
        import os as _os
        import tempfile
        history = [self._record("2026-08-31"), self._record("2026-09-01"),
                   self._record("2026-09-02")]
        with tempfile.TemporaryDirectory() as tmp:
            archive.archive_resolved(history, tmp)
            files = sorted(_os.listdir(tmp))
            self.assertEqual(files, ["2026-08.jsonl", "2026-09.jsonl"])
            loaded = archive.load_archive(tmp)
            self.assertEqual([r["session_date"] for r in loaded],
                             ["2026-08-31", "2026-09-01", "2026-09-02"])

    def test_a_corrupt_line_does_not_block_the_archive(self):
        import os as _os
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = _os.path.join(tmp, "2026-09.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("{broken json\n")
            self.assertEqual(archive.archive_resolved([self._record("2026-09-14")], tmp), 1)
            self.assertEqual(len(archive.load_archive(tmp)), 1)

    def test_csv_export_has_a_header_and_one_row_per_signal(self):
        rows = archive.to_csv([self._record("2026-09-14"),
                               self._record("2026-09-15", resolved=False)])
        lines = rows.strip().split("\n")
        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[0].startswith("session_date,generated_at,spot"))
        self.assertIn("25050.0", lines[1])
        self.assertIn("22.5", lines[1])

    def test_csv_round_trips_through_a_reader(self):
        import csv as _csv
        import io as _io
        text = archive.to_csv([self._record("2026-09-14")])
        parsed = list(_csv.DictReader(_io.StringIO(text)))
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["session_date"], "2026-09-14")
        self.assertEqual(parsed[0]["option_type"], "CE")
        self.assertEqual(parsed[0]["bias_correct"], "True")

    def test_merge_prefers_the_resolved_copy(self):
        archived = [self._record("2026-09-14")]
        live = [self._record("2026-09-14", resolved=False),
                self._record("2026-09-15", resolved=False)]
        merged = archive.merge_for_export(archived, live)
        self.assertEqual(len(merged), 2)
        self.assertIsNotNone(merged[0]["outcome"])   # kept the resolved one
        self.assertIsNone(merged[1]["outcome"])


class TestEndToEnd(unittest.TestCase):
    def test_full_replay_produces_a_coherent_model_and_scoreboard(self):
        cfg = load_config()
        candles = synthetic_candles(300, seed=21)
        result = replay(candles, synthetic_vix(candles, 21), cfg, warmup=240)
        history, perf, state = result["history"], result["performance"], result["state"]

        self.assertGreater(len(history), 30)
        self.assertGreater(perf["scored"], 10)
        # The replay feeds every session its follow-up close, so nothing is left
        # open; in production the newest signal stays pending until the next day.
        self.assertEqual(perf["pending"], 0)
        self.assertGreater(state["resolved"], 10)

        # Accuracy is a real percentage and every action is one of three values.
        self.assertTrue(0.0 <= perf["accuracy_all"] <= 100.0)
        for record in history:
            self.assertIn(record["action"], ("CALL", "PUT", "NO_TRADE"))
            self.assertIn(record["bias"], ("CALL", "PUT"))
            self.assertTrue(-1.0 <= record["score"] <= 1.0)
            self.assertTrue(0.0 <= record["probability_up"] <= 1.0)
            if record["action"] == "NO_TRADE":
                self.assertIsNone(record["trade"])
            else:
                self.assertIn(record["trade"]["option_type"], ("CE", "PE"))

        # The model actually learned something: weights are no longer uniform.
        names = list(history[-1]["votes"].keys())
        weights = learner.strategy_weights(state, "mid_vol", names, cfg["learning"])
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)
        self.assertGreater(max(weights.values()) - min(weights.values()), 0.01)
        self.assertGreater(len(state["calibration"]), 1)

    def test_published_payload_is_json_serialisable(self):
        import json
        cfg = load_config()
        candles = synthetic_candles(280, seed=31)
        result = replay(candles, synthetic_vix(candles, 31), cfg, warmup=260)
        json.dumps(result["history"], default=str)
        json.dumps(result["performance"], default=str)
        json.dumps(result["state"], default=str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
