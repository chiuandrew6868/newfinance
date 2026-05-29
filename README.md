# Finance Strategy Trend Analyzer

這個 Streamlit app 已改為手動上傳 Google Trends CSV 後才執行分析，不會自動連線抓取 Google Trends，也不需要 `pytrends`。

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
5. Deploy 後網站會以手動上傳 CSV 的方式運作，不會自動抓取 Google Trends。

## CSV 格式

可直接使用 Google Trends 匯出的 `Interest over time` CSV。也支援一般 CSV：

```csv
date,AI finance,ETF,interest rate
2026-01-01,42,56,81
2026-01-08,48,52,77
2026-01-15,55,63,74
```

第一欄需是日期欄，欄名可為 `date`、`day`、`week`、`month` 或 `time`；後續欄位會被視為關鍵字或主題。
