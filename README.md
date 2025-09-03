# O'Neil Short Scanner

日本株市場でウィリアム・オニールの空売りパターン（空売りダイアグラム）を検出するPythonスクリプトです。

## 概要

このスクリプトは、J-Quants APIを使用して日本の上場株式の日次価格データを分析し、O'Neilの空売り手法で知られる特定のパターンを自動検出します。

検出パターンの特徴：
- 十分な上昇トレンド後の高値形成
- 50日移動平均線を下回る大商量でのブレイクダウン
- 移動平均線付近での複数回の戻り売り失敗
- ヘッドアンドショルダーズ様のネックライン割れ

## 必要な環境

- Python 3.7+
- 必要なライブラリ：
  ```bash
  pip install jquantsapi pandas numpy
  ```

## セットアップ

1. J-Quants APIのトークンを取得
2. 環境変数にトークンを設定：
   ```bash
   export JQUANTS_TOKEN="your_token_here"
   ```

## 使用方法

```bash
python oneil_short_scanner.py
```

実行すると：
- 過去2年間の日次データを取得
- 全市場（プライム・スタンダード・グロース）をスキャン
- 検出結果を `oneil_short_candidates.csv` に保存

## 出力ファイル

`oneil_short_candidates.csv` には以下の情報が含まれます：
- `Code`: 銘柄コード
- `Market`: 市場区分
- `B_date`: 高値（ヘッド）の日付
- `break_date`: MA割れ日付
- `bounce_count`: 戻り売り回数
- `entry_dates`: エントリーシグナル日付
- `neckline`: ネックライン水準
- `score`: パターンスコア

## パラメータのカスタマイズ

`detect_oneil_short_candidates` 関数で以下のパラメータを調整可能：
- `uptrend_weeks`: 最小上昇トレンド期間（デフォルト：8週）
- `uptrend_gain`: 最小上昇率（デフォルト：25%）
- `ma_window`: 移動平均の期間（デフォルト：50日）
- `vol_mult`: ブレイクダウン時の出来高倍率（デフォルト：1.5倍）
- `target_bounce_range`: 戻り売り回数の範囲（デフォルト：2-4回）

## 注意事項

- J-Quants APIの利用制限に注意してください
- 投資判断は自己責任で行ってください
- このツールは教育・研究目的で提供されています