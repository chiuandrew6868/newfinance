# Finance Strategy Trend Analyzer

這個 Streamlit app 使用手動上傳 CSV 的方式運作，不會自動連線抓取 Google Trends 或股票價格資料，也不需要 `pytrends`。

## 功能

- 上傳 Google Trends CSV，分析關鍵字熱度、動能、波動與策略優先級。
- 上傳股票價格 CSV，使用 MACD + RSI 產生買賣訊號並進行回測。
- 顯示策略總報酬、買進持有報酬、最大回撤、Sharpe、交易次數與交易紀錄。
- 可下載趨勢摘要與回測明細 CSV。

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
5. Deploy 後網站會以手動上傳 CSV 的方式運作。

## Google Trends CSV 格式

可直接使用 Google Trends 匯出的 `Interest over time` CSV。也支援一般 CSV：

```csv
date,AI finance,ETF,interest rate
2026-01-01,42,56,81
2026-01-08,48,52,77
2026-01-15,55,63,74
```

第一欄需是日期欄，欄名可為 `date`、`day`、`week`、`month` 或 `time`；後續欄位會被視為關鍵字或主題。

## 股票價格 CSV 格式

股票回測至少需要 `date` 與 `close` 欄位，也可包含 `open`、`high`、`low`、`volume`。

```csv
date,open,high,low,close,volume
2026-01-02,100,103,99,102,1200000
2026-01-03,102,105,101,104,980000
2026-01-04,104,106,103,105,1130000
```
