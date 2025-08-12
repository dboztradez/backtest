# EURUSD Backtest Codespaces

This repository sets up a backtest environment for DR/IDR + ORB Forex strategies on EURUSD (M5).

## Quickstart
1. Upload this ZIP to your GitHub repository.
2. Open GitHub Codespaces on the repo.
3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the backtest (AlphaVantage example):
   ```bash
   export ALPHAVANTAGE_API_KEY=2WZRM91BFT2JMAZB
   python backtest.py --from 2023-08-12 --to 2025-08-12 --source alpha
   ```
5. Output:
   - `results/backtest_summary.json` → performance summary
   - `results/trades.csv` → trade-by-trade log
   - `results/equity_curve.csv` → equity over time

Replace `alpha` with `twelve`, `oanda`, or `csv` for other sources.
