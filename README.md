# Finance Strategy Backtester

這是一個 Streamlit 單頁網站。使用者輸入股票代碼後，系統會從 Yahoo Finance 取得股價資料，並可勾選 Google Trends、MACD、RSI 三種指標中的任意組合進行回測。

## 功能

- 股票代碼輸入，例如 `AAPL`、`TSLA`、`2330.TW`
- Yahoo Finance 自動取得股價資料
- Google Trends 維持手動上傳 CSV，不自動抓取 Google Trends
- 可勾選使用 Google Trends、MACD、RSI
- 顯示策略報酬、買進持有報酬、年化報酬、最大回撤、Sharpe、交易次數
- 顯示價格訊號、資金曲線、MACD、RSI、Google Trends 圖表

## 執行方式

```powershell
pip install -r requirements.txt
streamlit run app.py
```

## 部署到 Streamlit Community Cloud

1. 將本資料夾上傳到 GitHub repository。
2. 到 Streamlit Community Cloud 建立新 app。
3. Repository 選擇你的 GitHub repo。
4. Main file path 填入 `app.py`。
5. Deploy。

## Google Trends CSV 格式

可直接使用 Google Trends 匯出的 `Interest over time` CSV。也支援一般 CSV：

```csv
date,AI finance,ETF,interest rate
2026-01-01,42,56,81
2026-01-08,48,52,77
2026-01-15,55,63,74
```

第一欄需是日期欄，欄名可為 `date`、`day`、`week`、`month` 或 `time`；後續欄位會被視為關鍵字或主題。
