"""
ウィリアム・オニール空売りセットアップのトレンドスキャナー
===========================================

J-Quants APIを使用して日本株の「空売りダイアグラム」パターン
（ウィリアム・オニールが提唱した空売り手法）を検出するスクリプトです。
オニールの図表で示された構造をプログラム的に再現します：
疲弊した上昇トレンド、50日移動平均線を下回る大商量でのブレイク、
50日線付近での複数回の戻り失敗、ネックラインを下回る最終的なブレイクダウン。

コア検出ロジックは ``detect_oneil_short_candidates`` 関数に実装されています。
全上場日本株に対してフルスキャンを実行するには、J-Quants APIトークンを
``JQUANTS_TOKEN`` 環境変数に設定し、このファイルを実行してください。
結果は現在のディレクトリの ``oneil_short_candidates.csv`` に書き込まれます。

使用例::

    export JQUANTS_TOKEN=«your_token_here»
    python oneil_short_scanner.py

スキャンは全市場セグメント（プライム、スタンダード、グロース）をカバーし、
過去約2年間の日次価格データを考慮します。トレンドの強さ、出来高拡大、
戻り回数、ヘッドアンドショルダー類似度の閾値は関数引数で調整可能です。

注意: このスクリプトは `jquantsapi` パッケージに依存しています。
``pip install jquantsapi pandas numpy`` でインストールしてください。
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
    """*n*期間における平均真の値幅（ATR）を計算します。

    ATRは価格変動を正規化して閾値がボラティリティに応じてスケールするように
    使用されます。真の値幅（TR）の移動平均として定義され、TRは以下の最大値です：
    現在の高値-安値、現在の高値と前日終値の絶対差、現在の安値と前日終値の絶対差。
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
    """ローリング線形回帰の傾き（β）を計算します。

    長さ ``window`` の各ウィンドウについて、系列を0, 1, 2, …の系列に
    回帰させて傾きを返します。正の傾きは上昇トレンドを示します。
    傾きは1バーあたりの価格単位で表現されます。
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
    """系列内の局所的な極大・極小を特定します。

    点 ``i`` は、ウィンドウ ``[i-left, i+right]`` 内の値の最大値と
    等しい場合に局所最大と見なされ、同様に、そのウィンドウの最小値と
    等しい場合に局所最小と見なされます。
    局所高値と安値のインデックスが返されます。
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
    """単一株式の履歴内でオニール式空売り候補を検出します。

    パラメータ
    ----------
    df : DataFrame
        最低限 ``Date``, ``Open``, ``High``, ``Low``, ``Close``, ``Volume``
        列を含むDataFrame。データは日付昇順でソートされている必要があります。
        追加の列は無視されます。
    uptrend_weeks : int, デフォルト 8
        先行上昇トレンドの最小週数。この期間で株価は最低 ``uptrend_gain``
        上昇し、50日MA傾きが正である必要があります。
    uptrend_gain : float, デフォルト 0.25
        上昇トレンドウィンドウでの最小上昇率（割合）。
    ma_window : int, デフォルト 50
        主要サポートラインを定義する移動平均ウィンドウ。
    vol_mult : float, デフォルト 1.5
        ブレイクダウン日の出来高が ``ma_window`` 期間平均出来高を
        超える倍率。
    drop_from_peak_min : float, デフォルト 0.10
        最近のスイング高値(B)から移動平均下回りブレイクダウン（図の②）
        への最小下落率。
    drop_days_max : int, デフォルト 15
        高値(B)と初回ブレイクダウン間の最大営業日数。
        初回下落の速度を制御します。
    ma_band_eps : float, デフォルト 0.01
        戻りがMAに触れたと見なす相対的許容範囲。
        MAの ``1+eps`` 以内の戻り高値がカウントされます。
    min_bounce_atr : float, デフォルト 0.5
        ATRの倍数で表現された最小戻り幅。ノイズをフィルタリングします。
    target_bounce_range : tuple[int, int], デフォルト (2, 4)
        移動平均下での失敗戻り回数の許容範囲。
        2-4回戻りのパターンが好まれます。
    hs_tolerance : float, デフォルト 0.12
        ヘッドアンドショルダー類似度の許容範囲。左肩(A)と右肩(C)の高さは
        ヘッド(B)高さの ``±tolerance`` 以内である必要があります。

    戻り値
    -------
    List[dict]
        候補パターンのリスト。各エントリにはスイング高値の日付(``B_date``)、
        ブレイクダウン日付(``break_date``)、戻り回数(``bounce_count``)、
        エントリーシグナル(``entry_dates``)、ネックライン推定値(``neckline``)、
        総合ヒューリスティック``score``が含まれます。
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
    """指定された市場セグメントの全株式でオニールパターンをスキャンします。

    パラメータ
    ----------
    jq : JQuantsApi
        認証済みJQuantsApiクライアント。
    from_date, to_date : str
        日次株価を取得する包含的日付範囲を定義するISO形式文字列。
        十分な履歴を捕捉するため最低2年間の範囲が推奨されます。
    markets : list of str, オプション
        含める市場コードまたは名前。``None``の場合、全市場がスキャンされます。
        典型的な値には ``'プライム'``, ``'スタンダード'``, ``'グロース'`` があります。
        APIが英語で市場コードを返す場合は適宜調整してください。
    max_candidates_per_code : int, デフォルト 1
        銘柄ごとに記録する上位パターンの最大数。すべてを記録するには
        ``None``に設定します。
    verbose : bool, デフォルト True
        ``True``の場合、進捗が標準出力に表示されます。

    戻り値
    -------
    DataFrame
        全銘柄で検出されたパターンのテーブル。各行には ``Code``, ``Market``
        および候補の詳細（日付、戻り回数、スコアなど）が含まれます。
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
    """コマンドライン実行のエントリーポイント。"""
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
