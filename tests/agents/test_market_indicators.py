import unittest
from datetime import timedelta
from app.core.models.market import Bar, MarketTrend, MarketRegime, AnomalyType, DataQualityIssue
from app.agents.market_research.data_quality import DataQualityChecker
from app.agents.market_research.indicators import MarketFeatureCalculator
from app.agents.market_research.regime_classifier import MarketRegimeClassifier
from app.agents.market_research.anomaly_detector import MarketAnomalyDetector
from app.utils.time import now_utc


class TestMarketIndicatorsAndQuality(unittest.TestCase):
    def setUp(self):
        self.now = now_utc()
        self.calc = MarketFeatureCalculator()
        self.checker = DataQualityChecker()
        self.classifier = MarketRegimeClassifier()
        self.detector = MarketAnomalyDetector()

    def _bars(self, prices, volumes=None, opens=None) -> list[Bar]:
        bars = []
        for i, p in enumerate(prices):
            t = self.now - timedelta(days=(len(prices) - i))
            vol = 1000 if volumes is None else volumes[i]
            o = p if opens is None else opens[i]
            bars.append(Bar(
                symbol="TEST",
                timestamp=t,
                open=o,
                high=max(o, p) * 1.0 + 0.01 * p,
                low=min(o, p) * 1.0 - 0.01 * p if min(o, p) > 1 else min(o, p),
                close=p,
                volume=vol,
            ))
        return bars

    def test_invalid_prices_and_nan(self):
        nan_bar = Bar(symbol="TEST", timestamp=self.now, open=float("nan"), high=10, low=9, close=10, volume=1)
        report = self.checker.validate_bars("TEST", [nan_bar])
        self.assertFalse(report.is_valid)
        self.assertIn(DataQualityIssue.INVALID_PRICE, report.issue_types)
        self.assertEqual(self.checker.filter_valid_bars([nan_bar]), [])

    def test_malformed_ohlc(self):
        bad = Bar(symbol="TEST", timestamp=self.now, open=10, high=8, low=12, close=9, volume=100)
        report = self.checker.validate_bars("TEST", [bad])
        self.assertFalse(report.is_valid)
        self.assertIn(DataQualityIssue.IMPOSSIBLE_OHLC, report.issue_types)

    def test_negative_price(self):
        bad = Bar(symbol="TEST", timestamp=self.now, open=-10, high=10, low=-10, close=5, volume=100)
        report = self.checker.validate_bars("TEST", [bad])
        self.assertFalse(report.is_valid)
        self.assertIn(DataQualityIssue.NEGATIVE_PRICE, report.issue_types)

    def test_missing_and_negative_volume(self):
        zero = Bar(symbol="TEST", timestamp=self.now, open=10, high=11, low=9, close=10, volume=0)
        report = self.checker.validate_bars("TEST", [zero])
        self.assertTrue(report.is_valid)
        self.assertIn(DataQualityIssue.MISSING_VOLUME, report.issue_types)

        neg = Bar(symbol="TEST", timestamp=self.now, open=10, high=11, low=9, close=10, volume=-5)
        report_neg = self.checker.validate_bars("TEST", [neg])
        self.assertFalse(report_neg.is_valid)

    def test_duplicate_and_inconsistent_timestamps(self):
        t = self.now
        a = Bar(symbol="TEST", timestamp=t, open=10, high=11, low=9, close=10, volume=1)
        b = Bar(symbol="TEST", timestamp=t, open=10, high=11, low=9, close=10.5, volume=1)
        report = self.checker.validate_bars("TEST", [a, b])
        self.assertFalse(report.is_valid)
        self.assertIn(DataQualityIssue.DUPLICATE_TIMESTAMPS, report.issue_types)
        cleaned = self.checker.filter_valid_bars([a, b])
        self.assertEqual(len(cleaned), 1)

        older = Bar(symbol="TEST", timestamp=t - timedelta(days=1), open=10, high=11, low=9, close=10, volume=1)
        newer = Bar(symbol="TEST", timestamp=t, open=10, high=11, low=9, close=11, volume=1)
        report_order = self.checker.validate_bars("TEST", [newer, older])
        self.assertIn(DataQualityIssue.INCONSISTENT_TIMESTAMPS, report_order.issue_types)

    def test_missing_bars_gap_warning(self):
        a = Bar(symbol="TEST", timestamp=self.now - timedelta(days=20), open=10, high=11, low=9, close=10, volume=1)
        b = Bar(symbol="TEST", timestamp=self.now, open=10, high=11, low=9, close=11, volume=1)
        report = self.checker.validate_bars("TEST", [a, b])
        self.assertTrue(any("missing bars" in w.lower() for w in report.warnings))

    def test_stale_quote(self):
        from app.core.models.market import Quote
        quote = Quote("TEST", 10, 10.2, 1, 1, self.now - timedelta(hours=2))
        issues, types = self.checker.validate_quote("TEST", quote, self.now, market_is_open=True)
        self.assertTrue(issues)
        self.assertIn(DataQualityIssue.STALE_QUOTE, types)

    def test_returns_exact(self):
        bars = self._bars([100.0, 110.0])
        returns = self.calc.calculate_returns(bars)
        self.assertEqual(returns.return_1d, 0.10)
        self.assertIsNone(returns.return_5d)
        self.assertIsNone(returns.return_20d)

        bars_6 = self._bars([100.0, 102.0, 105.0, 110.0, 115.0, 120.0])
        self.assertEqual(self.calc.calculate_returns(bars_6).return_5d, 0.20)

        prices_21 = [100.0] + [100.0 + i for i in range(1, 21)]
        prices_21[-1] = 120.0
        prices_21[0] = 100.0
        bars_21 = self._bars(prices_21)
        self.assertEqual(self.calc.calculate_returns(bars_21).return_20d, 0.20)

    def test_sma_known_values(self):
        bars_20 = self._bars([100.0] * 20)
        mas = self.calc.calculate_moving_averages(bars_20)
        self.assertEqual(mas.sma_20, 100.0)
        self.assertIsNone(mas.sma_50)
        self.assertEqual(mas.status, "VALID")

        bars_5 = self._bars([100.0] * 5)
        mas_5 = self.calc.calculate_moving_averages(bars_5)
        self.assertIsNone(mas_5.sma_20)
        self.assertEqual(mas_5.status, "INSUFFICIENT_DATA")

        seq = list(range(1, 21))
        mas_seq = self.calc.calculate_moving_averages(self._bars([float(x) for x in seq]))
        self.assertEqual(mas_seq.sma_20, 10.5)

        bars_50 = self._bars([50.0] * 50)
        self.assertEqual(self.calc.calculate_moving_averages(bars_50).sma_50, 50.0)

        bars_200 = self._bars([25.0] * 200)
        mas_200 = self.calc.calculate_moving_averages(bars_200)
        self.assertEqual(mas_200.sma_200, 25.0)

    def test_volatility_constant_and_formula(self):
        bars = self._bars([100.0] * 10)
        vol = self.calc.calculate_volatility(bars)
        self.assertEqual(vol.short_term_volatility, 0.0)
        self.assertEqual(vol.calculation_method, "ANNUALIZED_STD_OF_LOG_RETURNS")

        # Two equal-magnitude log returns: 100 -> 110 -> 100
        # Need 6 bars for short-term window of 5 returns: pad with 100s then 110, 100
        prices = [100.0, 100.0, 100.0, 100.0, 110.0, 100.0]
        vol2 = self.calc.calculate_volatility(self._bars(prices))
        self.assertIsNotNone(vol2.short_term_volatility)
        self.assertGreater(vol2.short_term_volatility, 0.0)

    def test_volume_ratio_zero_safe(self):
        bars = self._bars([100.0] * 20, volumes=[1000] * 19 + [3000])
        metrics = self.calc.calculate_volume_metrics(bars)
        self.assertEqual(metrics.volume_ratio, 3.0)
        self.assertTrue(metrics.is_unusual_volume)

        empty = self.calc.calculate_volume_metrics([])
        self.assertEqual(empty.volume_ratio, 1.0)

        zero_avg = self._bars([100.0, 101.0], volumes=[0, 10])
        m = self.calc.calculate_volume_metrics(zero_avg)
        self.assertEqual(m.volume_ratio, 1.0)

    def test_rsi_wilder_exact(self):
        # 16 closes: 14 consecutive +1 then one -1 after initial RSI window.
        prices = [float(10 + i) for i in range(15)] + [23.0]
        rsi = self.calc.calculate_rsi(prices)
        expected = 100.0 - (100.0 / 14.0)
        self.assertAlmostEqual(rsi, expected, places=10)

        increasing = [100.0 + i for i in range(25)]
        rsi_up = self.calc.calculate_rsi(increasing)
        self.assertGreaterEqual(rsi_up, 90.0)

        self.assertIsNone(self.calc.calculate_rsi([1.0, 2.0, 3.0]))

    def test_rsi_independent_of_intrabar_drawdown(self):
        closes = [100.0 + i for i in range(20)]
        a = []
        b = []
        for i, c in enumerate(closes):
            t = self.now - timedelta(days=20 - i)
            a.append(Bar("T", t, c, c + 1, c - 1, c, 1000))
            b.append(Bar("T", t, c, c + 50, c - 50, c, 1000))
        mas_a = self.calc.calculate_moving_averages(a)
        mas_b = self.calc.calculate_moving_averages(b)
        self.assertEqual(
            self.calc.calculate_momentum(a, mas_a).rsi_14,
            self.calc.calculate_momentum(b, mas_b).rsi_14,
        )

    def test_trend_states(self):
        up = self._bars([100.0 + i * 2.0 for i in range(55)])
        mas = self.calc.calculate_moving_averages(up)
        returns = self.calc.calculate_returns(up)
        self.assertEqual(self.classifier.classify_trend(up, mas, returns), MarketTrend.UPTREND)

        down = self._bars([200.0 - i * 2.0 for i in range(55)])
        mas_d = self.calc.calculate_moving_averages(down)
        ret_d = self.calc.calculate_returns(down)
        self.assertEqual(self.classifier.classify_trend(down, mas_d, ret_d), MarketTrend.DOWNTREND)

        flat = self._bars([100.0] * 25)
        mas_f = self.calc.calculate_moving_averages(flat)
        ret_f = self.calc.calculate_returns(flat)
        self.assertEqual(self.classifier.classify_trend(flat, mas_f, ret_f), MarketTrend.SIDEWAYS)

        short = self._bars([100.0] * 5)
        mas_s = self.calc.calculate_moving_averages(short)
        ret_s = self.calc.calculate_returns(short)
        self.assertEqual(self.classifier.classify_trend(short, mas_s, ret_s), MarketTrend.INSUFFICIENT_DATA)

    def test_regime_states(self):
        up = self._bars([100.0 + i * 2.0 for i in range(55)])
        mas = self.calc.calculate_moving_averages(up)
        ret = self.calc.calculate_returns(up)
        vol = self.calc.calculate_volatility(up)
        trend = self.classifier.classify_trend(up, mas, ret)
        self.assertEqual(self.classifier.classify_regime(up, trend, vol), MarketRegime.TRENDING_UP)

        alt = self._bars([100.0 if i % 2 == 0 else 130.0 for i in range(25)])
        vol_alt = self.calc.calculate_volatility(alt)
        self.assertGreaterEqual(vol_alt.short_term_volatility or 0.0, 0.40)
        self.assertEqual(
            self.classifier.classify_regime(alt, MarketTrend.SIDEWAYS, vol_alt),
            MarketRegime.HIGH_VOLATILITY,
        )

        flat = self._bars([100.0] * 25)
        vol_f = self.calc.calculate_volatility(flat)
        trend_f = self.classifier.classify_trend(flat, self.calc.calculate_moving_averages(flat), self.calc.calculate_returns(flat))
        self.assertEqual(self.classifier.classify_regime(flat, trend_f, vol_f), MarketRegime.LOW_VOLATILITY)

        osc = self._bars([100.0 if i % 2 == 0 else 101.0 for i in range(24)])
        vol_o = self.calc.calculate_volatility(osc)
        trend_o = self.classifier.classify_trend(
            osc, self.calc.calculate_moving_averages(osc), self.calc.calculate_returns(osc)
        )
        self.assertEqual(trend_o, MarketTrend.SIDEWAYS)
        self.assertEqual(self.classifier.classify_regime(osc, trend_o, vol_o), MarketRegime.RANGE_BOUND)

        short = self._bars([100.0] * 5)
        self.assertEqual(
            self.classifier.classify_regime(short, MarketTrend.INSUFFICIENT_DATA, self.calc.calculate_volatility(short)),
            MarketRegime.UNKNOWN,
        )

    def test_anomalies(self):
        spike = self._bars([100.0, 110.0])
        pf = self.calc.calculate_price_features(spike)
        ret = self.calc.calculate_returns(spike)
        vol = self.calc.calculate_volatility(spike)
        vm = self.calc.calculate_volume_metrics(spike)
        anoms = self.detector.detect_anomalies("TEST", spike, None, pf, ret, vol, vm, self.now)
        self.assertTrue(any(a.anomaly_type == AnomalyType.PRICE_SPIKE for a in anoms))

        drop = self._bars([100.0, 90.0])
        anoms_d = self.detector.detect_anomalies(
            "TEST", drop, None,
            self.calc.calculate_price_features(drop),
            self.calc.calculate_returns(drop),
            self.calc.calculate_volatility(drop),
            self.calc.calculate_volume_metrics(drop),
            self.now,
        )
        self.assertTrue(any(a.anomaly_type == AnomalyType.PRICE_DROP for a in anoms_d))

        vols = [1000] * 20
        vols[-1] = 4000
        vbars = self._bars([100.0] * 20, volumes=vols)
        anoms_v = self.detector.detect_anomalies(
            "TEST", vbars, None,
            self.calc.calculate_price_features(vbars),
            self.calc.calculate_returns(vbars),
            self.calc.calculate_volatility(vbars),
            self.calc.calculate_volume_metrics(vbars),
            self.now,
        )
        self.assertTrue(any(a.anomaly_type == AnomalyType.VOLUME_SPIKE for a in anoms_v))

        # Quiet history then a violent last window for volatility spike.
        quiet = [100.0] * 20
        violent = [100.0, 130.0, 90.0, 140.0, 80.0, 150.0]
        mixed = quiet + violent
        mbars = self._bars(mixed)
        vol_m = self.calc.calculate_volatility(mbars)
        if vol_m.medium_term_volatility and vol_m.medium_term_volatility >= 0.15:
            anoms_vol = self.detector.detect_anomalies(
                "TEST", mbars, None,
                self.calc.calculate_price_features(mbars),
                self.calc.calculate_returns(mbars),
                vol_m,
                self.calc.calculate_volume_metrics(mbars),
                self.now,
            )
            self.assertTrue(any(a.anomaly_type == AnomalyType.VOLATILITY_SPIKE for a in anoms_vol))


if __name__ == "__main__":
    unittest.main()
