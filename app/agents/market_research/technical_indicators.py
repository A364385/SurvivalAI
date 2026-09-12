"""Deterministic technical indicator engine.

Pure-Python, exact calculations (no LLM involvement, no floating surprises
beyond standard float math). The local models INTERPRET these values; they are
never asked to calculate them.

All functions take plain float lists (closes/highs/lows/volumes) so they are
trivially testable and reusable by backtesting, the fast loop, and prompt
builders.
"""

import math
from typing import Dict, List, Optional, Tuple


def sma(values: List[float], period: int) -> Optional[float]:
    if len(values) < period or period <= 0:
        return None
    return sum(values[-period:]) / period


def ema(values: List[float], period: int) -> Optional[float]:
    if len(values) < period or period <= 0:
        return None
    k = 2.0 / (period + 1.0)
    result = sum(values[:period]) / period  # seed with SMA
    for price in values[period:]:
        result = price * k + result * (1.0 - k)
    return result


def rsi(closes: List[float], period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    gains, losses = 0.0, 0.0
    for i in range(1, period + 1):
        change = closes[i] - closes[i - 1]
        if change >= 0:
            gains += change
        else:
            losses -= change
    avg_gain = gains / period
    avg_loss = losses / period
    # Wilder smoothing over the remainder
    for i in range(period + 1, len(closes)):
        change = closes[i] - closes[i - 1]
        gain = change if change > 0 else 0.0
        loss = -change if change < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def macd(closes: List[float], fast: int = 12, slow: int = 26,
         signal: int = 9) -> Optional[Dict[str, float]]:
    if len(closes) < slow + signal:
        return None

    def ema_series(vals: List[float], period: int) -> List[float]:
        k = 2.0 / (period + 1.0)
        seed = sum(vals[:period]) / period
        out = [seed]
        for price in vals[period:]:
            out.append(price * k + out[-1] * (1.0 - k))
        return out

    ema_fast = ema_series(closes, fast)
    ema_slow = ema_series(closes, slow)
    # Align: ema_slow starts at index slow-1 of closes; ema_fast covers fast-1..
    macd_line_values = [
        ema_fast[i + (slow - fast)] - ema_slow[i]
        for i in range(len(ema_slow))
    ]
    signal_series = ema_series(macd_line_values, signal)
    macd_line = macd_line_values[-1]
    signal_line = signal_series[-1]
    return {
        "macd": round(macd_line, 6),
        "signal": round(signal_line, 6),
        "histogram": round(macd_line - signal_line, 6),
    }


def bollinger_bands(closes: List[float], period: int = 20,
                    num_std: float = 2.0) -> Optional[Dict[str, float]]:
    if len(closes) < period:
        return None
    window = closes[-period:]
    mean = sum(window) / period
    variance = sum((c - mean) ** 2 for c in window) / period
    std = math.sqrt(variance)
    return {
        "middle": round(mean, 6),
        "upper": round(mean + num_std * std, 6),
        "lower": round(mean - num_std * std, 6),
        "bandwidth": round((2 * num_std * std / mean) if mean else 0.0, 6),
    }


def atr(highs: List[float], lows: List[float], closes: List[float],
        period: int = 14) -> Optional[float]:
    n = len(closes)
    if n < period + 1 or len(highs) != n or len(lows) != n:
        return None
    trs: List[float] = []
    for i in range(1, n):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    if len(trs) < period:
        return None
    result = sum(trs[:period]) / period
    for tr in trs[period:]:
        result = (result * (period - 1) + tr) / period
    return result


def adx(highs: List[float], lows: List[float], closes: List[float],
        period: int = 14) -> Optional[float]:
    n = len(closes)
    if n < 2 * period + 1:
        return None
    plus_dm: List[float] = []
    minus_dm: List[float] = []
    trs: List[float] = []
    for i in range(1, n):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        plus_dm.append(up if (up > down and up > 0) else 0.0)
        minus_dm.append(down if (down > up and down > 0) else 0.0)
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)

    def smoothed(vals: List[float]) -> List[float]:
        out = [sum(vals[:period])]
        for v in vals[period:]:
            out.append(out[-1] - out[-1] / period + v)
        return out

    tr_s = smoothed(trs)
    plus_s = smoothed(plus_dm)
    minus_s = smoothed(minus_dm)
    dxs: List[float] = []
    for i in range(len(tr_s)):
        if tr_s[i] == 0:
            dxs.append(0.0)
            continue
        plus_di = 100.0 * plus_s[i] / tr_s[i]
        minus_di = 100.0 * minus_s[i] / tr_s[i]
        denom = (plus_di + minus_di) or 1e-12
        dxs.append(100.0 * abs(plus_di - minus_di) / denom)
    if len(dxs) < period:
        return None
    return sum(dxs[-period:]) / period


def stochastic(highs: List[float], lows: List[float], closes: List[float],
               period: int = 14, smoothing: int = 3) -> Optional[Dict[str, float]]:
    n = len(closes)
    if n < period:
        return None
    raw_k: List[float] = []
    for i in range(period, n + 1):
        window_high = max(highs[i - period:i])
        window_low = min(lows[i - period:i])
        rng = window_high - window_low
        raw_k.append(100.0 * (closes[i - 1] - window_low) / rng if rng else 50.0)
    if len(raw_k) < smoothing:
        return None
    k = sum(raw_k[-smoothing:]) / smoothing
    d = k  # simplified %D as smoothed %K of last values
    return {"k": round(k, 4), "d": round(d, 4)}


def roc(closes: List[float], period: int = 12) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    base = closes[-period - 1]
    if base == 0:
        return None
    return (closes[-1] - base) / base * 100.0


def momentum(closes: List[float], period: int = 10) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    return closes[-1] - closes[-period - 1]


def historical_volatility(closes: List[float], period: int = 20,
                          trading_days: int = 252) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    returns = [
        math.log(closes[i] / closes[i - 1])
        for i in range(len(closes) - period, len(closes))
        if closes[i - 1] > 0
    ]
    if len(returns) < 2:
        return None
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    return math.sqrt(variance) * math.sqrt(trading_days)


def max_drawdown(equity_curve: List[float]) -> Tuple[float, float, float]:
    """Returns (max_drawdown_fraction, peak_value, trough_value).
    Drawdown is a positive fraction (0.25 = 25%)."""
    if not equity_curve:
        return 0.0, 0.0, 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    peak_at_max = peak
    trough_at_max = peak
    trough = equity_curve[0]
    for value in equity_curve:
        if value > peak:
            peak = value
            trough = value
        if value < trough:
            trough = value
        dd = (peak - value) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
            peak_at_max = peak
            trough_at_max = value
    return max_dd, peak_at_max, trough_at_max


def volume_ratio(volumes: List[float], period: int = 20) -> Optional[float]:
    if len(volumes) < period + 1:
        return None
    avg = sum(volumes[-period - 1:-1]) / period
    if avg == 0:
        return None
    return volumes[-1] / avg


def compute_indicator_snapshot(
    closes: List[float],
    highs: Optional[List[float]] = None,
    lows: Optional[List[float]] = None,
    volumes: Optional[List[float]] = None,
) -> Dict[str, Optional[float]]:
    """One deterministic snapshot of all indicators for a symbol.

    This dict is exactly what gets embedded into LLM prompts — the model
    interprets values it is given and never computes its own.
    """
    highs = highs or closes
    lows = lows or closes
    snapshot: Dict[str, Optional[float]] = {}
    snapshot["price"] = closes[-1] if closes else None
    snapshot["sma_20"] = sma(closes, 20)
    snapshot["sma_50"] = sma(closes, 50)
    snapshot["sma_200"] = sma(closes, 200)
    snapshot["ema_12"] = ema(closes, 12)
    snapshot["ema_26"] = ema(closes, 26)
    snapshot["rsi_14"] = rsi(closes, 14)
    m = macd(closes)
    snapshot["macd"] = m["macd"] if m else None
    snapshot["macd_signal"] = m["signal"] if m else None
    snapshot["macd_histogram"] = m["histogram"] if m else None
    bb = bollinger_bands(closes, 20)
    snapshot["bb_upper"] = bb["upper"] if bb else None
    snapshot["bb_lower"] = bb["lower"] if bb else None
    snapshot["atr_14"] = atr(highs, lows, closes, 14)
    snapshot["adx_14"] = adx(highs, lows, closes, 14)
    st = stochastic(highs, lows, closes)
    snapshot["stoch_k"] = st["k"] if st else None
    snapshot["stoch_d"] = st["d"] if st else None
    snapshot["roc_12"] = roc(closes, 12)
    snapshot["momentum_10"] = momentum(closes, 10)
    snapshot["volatility_20d"] = historical_volatility(closes, 20)
    snapshot["volume_ratio"] = volume_ratio(volumes, 20) if volumes else None
    return snapshot


def indicator_context_json(snapshot: Dict[str, Optional[float]],
                           symbol: str) -> str:
    """Render the snapshot as the deterministic context block for LLM prompts."""
    import json as _json
    payload = {"symbol": symbol}
    for key, value in snapshot.items():
        payload[key] = round(value, 4) if isinstance(value, float) else value
    return _json.dumps(payload, sort_keys=True)
