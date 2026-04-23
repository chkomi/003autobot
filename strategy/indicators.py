"""
기술 지표 순수 함수 모음.
모든 함수는 pandas Series/DataFrame만 입출력하며 사이드 이펙트가 없다.
외부 의존성 없이 pandas/numpy로 직접 구현 (Python 3.14 완전 호환).
"""
import numpy as np
import pandas as pd


# ── EMA ──────────────────────────────────────────────────────────

def calc_ema(series: pd.Series, period: int) -> pd.Series:
    """지수 이동평균 (Wilder smoothing이 아닌 표준 EMA)"""
    return series.ewm(span=period, adjust=False).mean()


# ── RSI ──────────────────────────────────────────────────────────

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder smoothing)"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ── MACD ─────────────────────────────────────────────────────────

def calc_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD.
    Returns DataFrame with columns: macd, macd_hist, macd_signal
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame(
        {"macd": macd_line, "macd_hist": histogram, "macd_signal": signal_line},
        index=series.index,
    )


# ── Bollinger Bands ───────────────────────────────────────────────

def calc_bollinger(
    series: pd.Series,
    period: int = 20,
    std: float = 2.0,
) -> pd.DataFrame:
    """볼린저 밴드.
    Returns DataFrame with columns: bb_upper, bb_mid, bb_lower, bb_bandwidth, bb_pct
    """
    mid = series.rolling(window=period).mean()
    std_dev = series.rolling(window=period).std()
    upper = mid + std * std_dev
    lower = mid - std * std_dev
    bandwidth = (upper - lower) / mid.replace(0, np.nan)
    bb_pct = (series - lower) / (upper - lower).replace(0, np.nan)

    return pd.DataFrame(
        {
            "bb_upper": upper,
            "bb_mid": mid,
            "bb_lower": lower,
            "bb_bandwidth": bandwidth,
            "bb_pct": bb_pct,
        },
        index=series.index,
    )


# ── ATR ──────────────────────────────────────────────────────────

def calc_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average True Range (Wilder smoothing)"""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


# ── Supertrend ────────────────────────────────────────────────────

def calc_supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 10,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """슈퍼트렌드 지표.
    Returns DataFrame with columns:
        supertrend      : 슈퍼트렌드 값
        supertrend_dir  : 방향 (1=상승, -1=하락)
        supertrend_long : Long 레벨
        supertrend_short: Short 레벨
    numpy 배열 기반으로 구현해 pandas 3.x ChainedAssignment 경고 없음.
    """
    atr = calc_atr(high, low, close, period).values
    hl2 = ((high + low) / 2).values
    close_arr = close.values

    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr

    n = len(close_arr)
    upper_band = upper_basic.copy()
    lower_band = lower_basic.copy()
    supertrend = np.full(n, np.nan)
    direction = np.full(n, np.nan)

    for i in range(1, n):
        if np.isnan(atr[i]):
            continue

        # 상단 밴드 조정
        if upper_basic[i] < upper_band[i - 1] or close_arr[i - 1] > upper_band[i - 1]:
            upper_band[i] = upper_basic[i]
        else:
            upper_band[i] = upper_band[i - 1]

        # 하단 밴드 조정
        if lower_basic[i] > lower_band[i - 1] or close_arr[i - 1] < lower_band[i - 1]:
            lower_band[i] = lower_basic[i]
        else:
            lower_band[i] = lower_band[i - 1]

        # 방향 결정
        prev_st = supertrend[i - 1]
        if np.isnan(prev_st):
            supertrend[i] = upper_band[i]
            direction[i] = -1
        elif prev_st == upper_band[i - 1]:
            # 이전이 상단 밴드 (하락 추세)
            if close_arr[i] > upper_band[i]:
                supertrend[i] = lower_band[i]
                direction[i] = 1
            else:
                supertrend[i] = upper_band[i]
                direction[i] = -1
        else:
            # 이전이 하단 밴드 (상승 추세)
            if close_arr[i] < lower_band[i]:
                supertrend[i] = upper_band[i]
                direction[i] = -1
            else:
                supertrend[i] = lower_band[i]
                direction[i] = 1

    idx = close.index
    return pd.DataFrame(
        {
            "supertrend": supertrend,
            "supertrend_dir": direction,
            "supertrend_long": lower_band,
            "supertrend_short": upper_band,
        },
        index=idx,
    )


# ── 볼륨 이동평균 ─────────────────────────────────────────────────

def calc_volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
    """거래량 단순 이동평균"""
    return volume.rolling(window=period).mean()


# ── ADX (Average Directional Index) ──────────────────────────────

def calc_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.DataFrame:
    """ADX 지표.
    Returns DataFrame with columns: adx, plus_di, minus_di
    """
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    # True Range
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Directional Movement
    plus_dm = np.where((high - prev_high) > (prev_low - low), np.maximum(high - prev_high, 0), 0)
    minus_dm = np.where((prev_low - low) > (high - prev_high), np.maximum(prev_low - low, 0), 0)

    plus_dm = pd.Series(plus_dm, index=high.index, dtype=float)
    minus_dm = pd.Series(minus_dm, index=high.index, dtype=float)

    # Wilder smoothing
    atr = tr.ewm(com=period - 1, adjust=False).mean()
    smooth_plus = plus_dm.ewm(com=period - 1, adjust=False).mean()
    smooth_minus = minus_dm.ewm(com=period - 1, adjust=False).mean()

    plus_di = 100 * smooth_plus / atr.replace(0, np.nan)
    minus_di = 100 * smooth_minus / atr.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(com=period - 1, adjust=False).mean()

    return pd.DataFrame({
        "adx": adx,
        "plus_di": plus_di,
        "minus_di": minus_di,
    }, index=high.index)


# ── OBV (On-Balance Volume) ──────────────────────────────────────

def calc_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume"""
    direction = np.sign(close.diff())
    return (volume * direction).fillna(0).cumsum()


# ── Stochastic Oscillator ────────────────────────────────────────

def calc_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> pd.DataFrame:
    """스토캐스틱 오실레이터.
    Returns DataFrame with columns: stoch_k, stoch_d
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    stoch_k = 100 * (close - lowest_low) / denom
    stoch_d = stoch_k.rolling(window=d_period).mean()

    return pd.DataFrame({
        "stoch_k": stoch_k,
        "stoch_d": stoch_d,
    }, index=close.index)


# ── 편의 함수: MACD 크로스 감지 ──────────────────────────────────

def detect_macd_cross(macd_df: pd.DataFrame, lookback: int = 3) -> str:
    """최근 lookback 캔들 내 MACD 히스토그램 크로스 발생 감지.

    Args:
        lookback: 최근 N캔들 내 크로스를 탐색 (기본 3)

    Returns:
        "UP"   : 히스토그램 음→양 전환 발생
        "DOWN" : 히스토그램 양→음 전환 발생
        "NONE" : 크로스 없음
    """
    hist = macd_df["macd_hist"].dropna()
    if len(hist) < 2:
        return "NONE"

    n = min(lookback + 1, len(hist))
    recent = hist.iloc[-n:]

    for i in range(1, len(recent)):
        prev = recent.iloc[i - 1]
        curr = recent.iloc[i]
        if prev < 0 and curr >= 0:
            return "UP"
        if prev > 0 and curr <= 0:
            return "DOWN"
    return "NONE"
