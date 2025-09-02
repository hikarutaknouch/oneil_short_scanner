"""
Trend Scanner for William O'Neil Short‑Selling Setup
--------------------------------------------------

This script scans Japanese equities for the “空売りダイアグラム”
pattern (a short‑selling setup popularised by William O'Neil) using
daily price data from the J‑Quants API.  It reproduces, in a
programmatic way, the structure illustrated in O'Neil's diagram:
an exhausted uptrend, a high‑volume break below the 50‑day moving
average, followed by multiple failed rally attempts near the 50‑day
line and a subsequent breakdown beneath a neckline.

The core detection logic is implemented in the
``detect_oneil_short_candidates`` function.  To run a full scan across
all listed Japanese equities, set your J‑Quants API token in the
``JQUANTS_TOKEN`` environment variable and execute this file.  The
results are written to ``oneil_short_candidates.csv`` in the current
directory.

Example usage::

    export JQUANTS_TOKEN=«your_token_here»
    python jp_trend_scanner_scan.py

The scan covers all market segments (プライム、スタンダード、グロース) and
considers roughly the past two years of daily price data.  Thresholds
for trend strength, volume expansion, rebound count and head‑and‑
shoulders similarity can be customised via the function arguments.

Note: this script depends on the `jquantsapi` package.  Install it
with ``pip install jquantsapi pandas numpy``.
"""

from __future__ import annotations

import datetime
import os
from typing import List, Dict, Any

import numpy as np
import pandas as pd

try:
    from jquantsapi import JQuantsApi  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "The jquantsapi library is required. Please install it with "
        "`pip install jquantsapi pandas numpy` and try again."
    ) from exc


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Compute the Average True Range (ATR) over *n* periods.

    ATR is used here to normalise price swings so that thresholds scale
    with volatility.  It is defined as the rolling mean of the True
    Range (TR), where TR is the maximum of the current high minus low,
    the absolute difference between the current high and the previous
    close, and the absolute difference between the current low and
    previous close.
    """
    prev_close = df['Close'].shift(1)
    tr = pd.concat(
        [
            df['High'] - df['Low'],
            (df['High'] - prev_close).abs(),
            (df['Low'] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def rolling_slope(series: pd.Series, window: int = 50) -> pd.Series:
    """Compute the slope (β) of a rolling linear regression.

    For each window of length ``window``, the series is regressed on a
    sequence 0, 1, 2, … and the slope is returned.  Positive slopes
    indicate uptrends.  The slope is expressed in units of price per
    bar.
    """
    x = np.arange(window)

    def _slope(y: pd.Series) -> float:
        y_arr = np.array(y)
        x_mean = x.mean()
        y_mean = y_arr.mean()
        numerator = ((x - x_mean) * (y_arr - y_mean)).sum()
        denominator = ((x - x_mean) ** 2).sum()
        return numerator / denominator

    return series.rolling(window, min_periods=window).apply(_slope, raw=False)


def local_pivots(
    series: pd.Series, left: int = 3, right: int = 3
) -> tuple[List[int], List[int]]:
    """Identify local maxima and minima in a series.

    A point ``i`` is considered a local maximum if it equals the
    maximum of the values in the window ``[i-left, i+right]`` and
    similarly a local minimum if it equals the minimum of that window.
    The indices of local highs and lows are returned.
    """
    highs: List[int] = []
    lows: List[int] = []
    for i in range(left, len(series) - right):
        window = series.iloc[i - left : i + right + 1]
        if series.iloc[i] == window.max():
            highs.append(i)
        if series.iloc[i] == window.min():
            lows.append(i)
    return highs, lows


def detect_oneil_short_candidates(
    df: pd.DataFrame,
    uptrend_weeks: int = 8,
    uptrend_gain: float = 0.25,
    ma_window: int = 50,
    vol_mult: float = 1.5,
    drop_from_peak_min: float = 0.10,
    drop_days_max: int = 15,
    ma_band_eps: float = 0.01,
    min_bounce_atr: float = 0.5,
    target_bounce_range: tuple[int, int] = (2, 4),
    hs_tolerance: float = 0.12,
) -> List[Dict[str, Any]]:
    """Detect O'Neil-style short‑selling candidates within a single equity's history.

    Parameters
    ----------
    df : DataFrame
        A DataFrame containing columns ``Date``, ``Open``, ``High``, ``Low``,
        ``Close``, ``Volume`` at minimum.  The data must be sorted by date
        ascending.  Additional columns are ignored.
    uptrend_weeks : int, default 8
        Minimum number of weeks for the preceding uptrend.  The stock must
        have risen by at least ``uptrend_gain`` over this period and the
        50‑day MA slope must be positive.
    uptrend_gain : float, default 0.25
        Minimum gain (as a fraction) over the uptrend window.
    ma_window : int, default 50
        The moving average window used to define the key support line.
    vol_mult : float, default 1.5
        Factor by which volume on the breakdown day must exceed the
        ``ma_window``‑period average volume.
    drop_from_peak_min : float, default 0.10
        Minimum fractional drop from the recent swing high (B) to the
        breakdown below the moving average (② in the diagram).
    drop_days_max : int, default 15
        Maximum number of trading days between the high (B) and the initial
        breakdown.  Controls how fast the first decline occurs.
    ma_band_eps : float, default 0.01
        Relative tolerance for considering a rebound as touching the MA.
        A rebound high within ``1+eps`` of the MA is counted.
    min_bounce_atr : float, default 0.5
        Minimum rebound amplitude expressed in multiples of ATR.  Filters
        out noise.
    target_bounce_range : tuple[int, int], default (2, 4)
        Acceptable range for the number of failed rebounds underneath the
        moving average.  Patterns with 2–4 rebounds are favoured.
    hs_tolerance : float, default 0.12
        Tolerance for head‑and‑shoulders similarity.  The heights of the
        left shoulder (A) and right shoulder (C) should be within
        ``±tolerance`` of the head (B) height.

    Returns
    -------
    List[dict]
        A list of candidate patterns.  Each entry contains the dates of
        the swing high (``B_date``), breakdown (``break_date``), number of
        rebounds (``bounce_count``), entry signals (``entry_dates``),
        neckline estimate (``neckline``) and an overall heuristic
        ``score``.
    """
    d = df.copy().reset_index(drop=True)
    # Ensure required columns exist
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col not in d.columns:
            raise ValueError(f"missing required column: {col}")

    d['MA'] = d['Close'].rolling(ma_window).mean()
    d['VOL_MA'] = d['Volume'].rolling(ma_window).mean()
    d['ATR'] = atr(d, 14)
    d['MA_slope'] = rolling_slope(d['MA'], window=ma_window)

    # Uptrend condition: positive slope and sufficient gain over prior period
    bars_up = uptrend_weeks * 5  # approximate number of trading days
    d['uptrend_ok'] = False
    if len(d) > bars_up:
        gains = d['Close'] / d['Close'].shift(bars_up) - 1
        d.loc[:, 'uptrend_ok'] = (
            (d['MA_slope'] > 0)
            & (gains >= uptrend_gain)
        )

    # Identify local swing highs and lows
    hi_idx, lo_idx = local_pivots(d['Close'], left=3, right=3)
    hi_idx_set = set(hi_idx)

    candidates: List[Dict[str, Any]] = []
    # Iterate over swing highs as potential heads (B)
    for b in hi_idx:
        # Skip early bars without full MA history or uptrend
        if b < ma_window + bars_up:
            continue
        if not bool(d.loc[b, 'uptrend_ok']):
            continue
        # Search for first breakdown below MA within drop_days_max
        search_end = min(len(d) - 1, b + drop_days_max)
        post_b = d.iloc[b + 1 : search_end + 1]
        # Condition: cross from above to below the moving average
        cross = post_b[
            (post_b['Close'] < post_b['MA'])
            & (post_b['Close'].shift(1) >= post_b['MA'].shift(1))
        ]
        if cross.empty:
            continue
        t_break = cross.index[0]
        # Volume expansion
        vol_ok = d.loc[t_break, 'Volume'] >= vol_mult * d.loc[t_break, 'VOL_MA']
        # Price drop magnitude
        drop_ok = (
            (d.loc[b, 'Close'] - d.loc[t_break, 'Close']) / d.loc[b, 'Close']
            >= drop_from_peak_min
        )
        if not (drop_ok and vol_ok):
            continue
        # Count failed rebounds under the MA over a 60‑bar horizon
        horizon = min(len(d) - 1, t_break + 60)
        seg = d.iloc[t_break + 1 : horizon + 1]
        pivot_highs, _ = local_pivots(seg['Close'], left=2, right=2)
        bounce_idx: List[int] = []
        # Track the previous low to measure rebound amplitude
        for i_rel in pivot_highs:
            idx = seg.index[i_rel]
            if d.loc[idx, 'Close'] <= (1 + ma_band_eps) * d.loc[idx, 'MA']:
                # Identify the nearest prior local low between t_break and this high
                prior_lows = [j for j in lo_idx if t_break < j < idx]
                if prior_lows:
                    prev_low = max(prior_lows)
                    amp = d.loc[idx, 'Close'] - d.loc[prev_low, 'Close']
                    if amp >= min_bounce_atr * d.loc[idx, 'ATR']:
                        bounce_idx.append(idx)
        bounce_count = len(bounce_idx)
        # Compute H&S similarity (optional score boost)
        a_candidates = [i for i in hi_idx if i < b][-3:]
        c_candidates = bounce_idx[-3:]
        hs_score = 0
        neck_level = np.nan
        if a_candidates and c_candidates:
            A = a_candidates[-1]
            C = c_candidates[-1]
            hA = d.loc[A, 'Close'] / d.loc[b, 'Close']
            hC = d.loc[C, 'Close'] / d.loc[b, 'Close']
            if (
                (1 - hs_tolerance) <= hA <= 1 + hs_tolerance
                and (1 - hs_tolerance) <= hC <= 1 + hs_tolerance
            ):
                left_neck = [i for i in lo_idx if A < i < b]
                right_neck = [i for i in lo_idx if b < i < C]
                if left_neck and right_neck:
                    neck_level = (
                        d.loc[left_neck, 'Close'].min()
                        + d.loc[right_neck, 'Close'].min()
                    ) / 2
                    hs_score = 1
        # Determine entry signals: day after failed rebound or neckline break
        entry_days: List[int] = []
        # Failed rebound: look for a bearish day after the rebound high
        for idx in bounce_idx:
            if (
                d.loc[idx, 'Close'] < d.loc[idx, 'Open']
                and d.loc[idx + 1, 'Close'] < d.loc[idx, 'Close']
            ):
                entry_days.append(idx + 1)
        # Neckline break: first close below neck level
        if not np.isnan(neck_level):
            post_seg = d.iloc[t_break + 1 : horizon + 1]
            neck_break = post_seg[post_seg['Close'] < neck_level]
            if not neck_break.empty:
                entry_days.append(neck_break.index[0])
        # Score construction: base points plus optional enhancements
        score = 0
        score += 1  # Uptrend requirement
        score += 1  # Breakdown below MA
        score += int(vol_ok)  # Volume expansion
        score += int(
            target_bounce_range[0] <= bounce_count <= target_bounce_range[1]
        )  # Rebound count
        score += hs_score
        # Append candidate
        candidates.append(
            {
                'B_date': d.loc[b, 'Date'] if 'Date' in d.columns else b,
                'break_date': d.loc[t_break, 'Date'] if 'Date' in d.columns else t_break,
                'bounce_count': bounce_count,
                'entry_dates': [
                    d.loc[x, 'Date'] if 'Date' in d.columns else x
                    for x in sorted(set(entry_days))
                ],
                'neckline': float(neck_level) if not np.isnan(neck_level) else None,
                'score': score,
            }
        )
    # Sort by score descending and earliest break_date ascending
    candidates.sort(key=lambda x: (-x['score'], x['break_date']))
    return candidates


def scan_all_equities(
    jq: JQuantsApi,
    from_date: str,
    to_date: str,
    markets: List[str] | None = None,
    max_candidates_per_code: int = 1,
    verbose: bool = True,
) -> pd.DataFrame:
    """Scan all equities in the specified market segments for O'Neil patterns.

    Parameters
    ----------
    jq : JQuantsApi
        An authenticated JQuantsApi client.
    from_date, to_date : str
        ISO format strings defining the inclusive date range to fetch daily
        quotes.  A range of at least two years is recommended to capture
        sufficient history.
    markets : list of str, optional
        Market codes or names to include.  If ``None``, all markets are
        scanned.  Typical values include ``'プライム'``, ``'スタンダード'`` and
        ``'グロース'``.  If the API returns market codes in English, adjust
        accordingly.
    max_candidates_per_code : int, default 1
        Maximum number of top patterns to record per security.  Set to
        ``None`` to record all.
    verbose : bool, default True
        If ``True``, progress is printed to stdout.

    Returns
    -------
    DataFrame
        A table of detected patterns across all securities.  Each row
        contains ``Code``, ``Market`` and details of a candidate (dates,
        bounce count, score, etc.).
    """
    # Retrieve listing info to obtain codes and market categories
    listing = jq.get_listed_info()
    if not isinstance(listing, list):
        # Newer versions may return dict
        listing_df = pd.DataFrame(listing)
    else:
        listing_df = pd.DataFrame(listing)
    # Filter by markets if requested
    if markets:
        # Normalise column names; JQuants uses both 'MarketCode' and
        # 'Market' depending on endpoint.  We support both.
        market_col = None
        for col in ['MarketCode', 'Market', 'marketCode', 'market']:
            if col in listing_df.columns:
                market_col = col
                break
        if market_col:
            listing_df = listing_df[listing_df[market_col].isin(markets)]
    codes = listing_df['Code'].unique().tolist()
    # Fetch daily quotes in bulk for the specified period.  When code is
    # ``None``, JQuants returns all securities in the universe.
    if verbose:
        print(
            f"Fetching daily quotes from {from_date} to {to_date} for {len(codes)} codes…"
        )
    daily_quotes = jq.get_prices_daily_quotes(
        code=None, from_=from_date, to_=to_date
    )
    quotes_df = pd.DataFrame(daily_quotes)
    # Convert date strings to datetime
    quotes_df['Date'] = pd.to_datetime(quotes_df['Date'])
    # Sort for each security
    quotes_df.sort_values(['Code', 'Date'], inplace=True)
    # Prepare results container
    rows: List[Dict[str, Any]] = []
    # Group by code
    for code, group in quotes_df.groupby('Code'):
        if code not in codes:
            continue  # skip codes not in desired markets
        # Compute pattern detection
        candidates = detect_oneil_short_candidates(group)
        if candidates:
            take = candidates if max_candidates_per_code is None else candidates[:max_candidates_per_code]
            for cand in take:
                row = {'Code': code}
                # Add market information
                market_info = listing_df[listing_df['Code'] == code]
                row['Market'] = (
                    market_info.iloc[0][market_col]
                    if markets is not None and market_col
                    else None
                )
                # Merge candidate fields
                row.update(cand)
                rows.append(row)
        if verbose:
            print(f"Scanned {code}: {len(candidates)} candidate(s)")
    return pd.DataFrame(rows)


def main() -> None:
    """Entry point for command‑line execution."""
    token = os.getenv('JQUANTS_TOKEN')
    if not token:
        raise SystemExit(
            'Set the JQUANTS_TOKEN environment variable with your API token.'
        )
    jq = JQuantsApi(token=token)
    # Define date range: last two years up to today
    today = datetime.date.today()
    from_date = (today - datetime.timedelta(days=730)).isoformat()
    to_date = today.isoformat()
    # Define all markets: None means all markets
    markets = None
    results = scan_all_equities(
        jq=jq,
        from_date=from_date,
        to_date=to_date,
        markets=markets,
        max_candidates_per_code=1,
        verbose=True,
    )
    # Save results to CSV
    if not results.empty:
        results.to_csv('oneil_short_candidates.csv', index=False)
        print(
            f"Saved {len(results)} candidate rows to oneil_short_candidates.csv"
        )
    else:
        print("No candidates detected in the specified period.")


if __name__ == '__main__':
    main()
