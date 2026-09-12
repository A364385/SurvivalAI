"""Tests for the deterministic technical indicator engine."""

import math
import unittest

from app.agents.market_research.technical_indicators import (
    adx, atr, bollinger_bands, compute_indicator_snapshot, ema,
    historical_volatility, indicator_context_json, macd, max_drawdown,
    momentum, roc, rsi, sma, stochastic, volume_ratio,
)


def _series(values):
    return list(values)


class TestMovingAverages(unittest.TestCase):
    def test_sma_exact(self):
        self.assertEqual(sma([1, 2, 3, 4, 5], 5), 3.0)
        self.assertEqual(sma([10, 20, 30], 2), 25.0)

    def test_sma_insufficient_data(self):
        self.assertIsNone(sma([1, 2], 5))

    def test_ema_seed_and_step(self):
        # Seed = SMA of first 3 = 2.0; then k=0.5 -> (4*0.5 + 2*0.5) = 3.0
        self.assertEqual(ema([1, 2, 3, 4], 3), 3.0)

    def test_ema_insufficient(self):
        self.assertIsNone(ema([1], 3))


class TestOscillators(unittest.TestCase):
    def test_rsi_all_gains_is_100(self):
        closes = [float(i) for i in range(1, 30)]
        self.assertEqual(rsi(closes), 100.0)

    def test_rsi_all_losses_is_near_zero(self):
        closes = [float(30 - i) for i in range(30)]
        value = rsi(closes)
        self.assertIsNotNone(value)
        self.assertLessEqual(value, 100 - 99.0 + 0.01)  # RS -> 0 => RSI -> 0

    def test_rsi_neutral_flat(self):
        closes = [50.0] * 30
        self.assertEqual(rsi(closes), 50.0)

    def test_rsi_bounds(self):
        closes = [50 + math.sin(i / 3.0) * 5 for i in range(60)]
        value = rsi(closes)
        self.assertTrue(0.0 <= value <= 100.0)

    def test_macd_shape(self):
        closes = [100 + math.sin(i / 5.0) * 10 for i in range(60)]
        result = macd(closes)
        self.assertIsNotNone(result)
        self.assertIn("macd", result)
        self.assertIn("signal", result)
        self.assertIn("histogram", result)
        self.assertAlmostEqual(
            result["histogram"], result["macd"] - result["signal"], places=5
        )

    def test_stochastic_bounds(self):
        highs = [110.0] * 20
        lows = [90.0] * 20
        closes = [95.0, 100.0, 105.0] * 7
        result = stochastic(highs, lows, closes[:20])
        self.assertIsNotNone(result)
        self.assertTrue(0 <= result["k"] <= 100)


class TestBandsAndVolatility(unittest.TestCase):
    def test_bollinger_symmetric(self):
        closes = [100.0] * 20
        bb = bollinger_bands(closes, 20, 2.0)
        self.assertEqual(bb["upper"], 100.0)
        self.assertEqual(bb["lower"], 100.0)

    def test_bollinger_constant_distance(self):
        closes = [100.0] * 10 + [110.0] * 10
        bb = bollinger_bands(closes, 20, 2.0)
        self.assertAlmostEqual(bb["upper"] - bb["middle"], bb["middle"] - bb["lower"], places=6)

    def test_atr_positive(self):
        highs = [110.0] * 20
        lows = [90.0] * 20
        closes = [100.0] * 20
        value = atr(highs, lows, closes, 14)
        self.assertEqual(value, 20.0)

    def test_historical_volatility_zero_for_flat(self):
        closes = [100.0] * 25
        self.assertEqual(historical_volatility(closes), 0.0)

    def test_historical_volatility_positive_for_moves(self):
        closes = [100 + (i % 2) * 10 for i in range(25)]
        self.assertGreater(historical_volatility(closes), 0.0)


class TestStrengthAndMomentum(unittest.TestCase):
    def test_adx_reasonable_bounds(self):
        closes = [100 + math.sin(i / 4.0) * 5 for i in range(60)]
        highs = [c + 2 for c in closes]
        lows = [c - 2 for c in closes]
        value = adx(highs, lows, closes)
        self.assertIsNotNone(value)
        self.assertTrue(0.0 <= value <= 100.0)

    def test_roc_exact(self):
        closes = [100.0, 110.0]
        self.assertEqual(roc(closes, 1), 10.0)

    def test_momentum_exact(self):
        self.assertEqual(momentum([10, 11, 12, 15], 3), 5.0)

    def test_volume_ratio_exact(self):
        volumes = [100.0] * 20 + [200.0]
        self.assertEqual(volume_ratio(volumes), 2.0)


class TestDrawdown(unittest.TestCase):
    def test_max_drawdown_exact(self):
        dd, peak, trough = max_drawdown([100, 120, 90, 95, 60, 110])
        self.assertAlmostEqual(dd, 0.5, places=9)
        self.assertEqual(peak, 120.0)
        self.assertEqual(trough, 60.0)

    def test_monotonic_growth_no_drawdown(self):
        dd, _, _ = max_drawdown([1, 2, 3, 4])
        self.assertEqual(dd, 0.0)

    def test_empty(self):
        self.assertEqual(max_drawdown([]), (0.0, 0.0, 0.0))


class TestSnapshot(unittest.TestCase):
    def test_snapshot_contains_all_core_indicators(self):
        closes = [100 + math.sin(i / 5.0) * 5 for i in range(220)]
        highs = [c + 1 for c in closes]
        lows = [c - 1 for c in closes]
        volumes = [1000.0 + i for i in range(220)]
        snap = compute_indicator_snapshot(closes, highs, lows, volumes)
        for key in ("price", "sma_20", "sma_50", "sma_200", "rsi_14", "macd",
                    "atr_14", "adx_14", "volatility_20d", "volume_ratio"):
            self.assertIn(key, snap)
        self.assertIsNotNone(snap["sma_200"])
        self.assertIsNotNone(snap["rsi_14"])

    def test_snapshot_handles_short_series(self):
        snap = compute_indicator_snapshot([100.0, 101.0])
        self.assertEqual(snap["price"], 101.0)
        self.assertIsNone(snap["sma_20"])

    def test_context_json_roundtrip(self):
        snap = compute_indicator_snapshot([100.0, 101.0, 102.0])
        text = indicator_context_json(snap, "AAPL")
        self.assertIn("AAPL", text)
        import json
        parsed = json.loads(text)
        self.assertEqual(parsed["symbol"], "AAPL")


if __name__ == "__main__":
    unittest.main()
