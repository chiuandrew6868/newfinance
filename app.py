import io
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


@dataclass
class TrendSummary:
    keyword: str
    latest: float
    previous: float
    average: float
    momentum: float
    volatility: float
    score: float
    recommendation: str


def find_header_row(raw_csv: bytes) -> int:
    preview = raw_csv.decode("utf-8-sig", errors="ignore").splitlines()
    date_labels = {"date", "day", "week", "month", "time"}

    for index, line in enumerate(preview[:30]):
        first_cell = line.split(",", 1)[0].strip().lower()
        if first_cell in date_labels:
            return index
    return 0


def load_trends_csv(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    header_row = find_header_row(raw)
    df = pd.read_csv(io.BytesIO(raw), skiprows=header_row)
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")

    if df.empty or len(df.columns) < 2:
        raise ValueError("CSV 至少需要一個日期欄位與一個關鍵字數值欄位。")

    date_column = df.columns[0]
    df = df.rename(columns={date_column: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    value_columns = [column for column in df.columns if column != "date"]
    for column in value_columns:
        df[column] = (
            df[column]
            .astype(str)
            .str.replace("<1", "0.5", regex=False)
            .str.replace(",", "", regex=False)
            .str.extract(r"([-+]?\d*\.?\d+)", expand=False)
        )
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df[["date"] + value_columns].dropna(axis=1, how="all")
    if len(df.columns) < 2:
        raise ValueError("找不到可分析的 Google Trends 數值欄位。")

    return df


def normalize(series: pd.Series) -> pd.Series:
    minimum = series.min()
    maximum = series.max()
    if pd.isna(minimum) or pd.isna(maximum) or maximum == minimum:
        return pd.Series(50, index=series.index, dtype=float)
    return (series - minimum) / (maximum - minimum) * 100


def build_summary(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    summaries: list[TrendSummary] = []

    for keyword in [column for column in df.columns if column != "date"]:
        series = df[keyword].dropna()
        if series.empty:
            continue

        window = series.tail(max(2, lookback))
        latest = float(window.iloc[-1])
        previous = float(window.iloc[-2]) if len(window) >= 2 else latest
        average = float(window.mean())
        momentum = latest - average
        volatility = float(window.std(ddof=0)) if len(window) > 1 else 0.0
        trend_strength = latest + momentum - volatility * 0.25

        if trend_strength >= 75:
            recommendation = "積極投入"
        elif trend_strength >= 55:
            recommendation = "加碼觀察"
        elif trend_strength >= 35:
            recommendation = "維持測試"
        else:
            recommendation = "降低優先"

        summaries.append(
            TrendSummary(
                keyword=keyword,
                latest=latest,
                previous=previous,
                average=average,
                momentum=momentum,
                volatility=volatility,
                score=trend_strength,
                recommendation=recommendation,
            )
        )

    summary_df = pd.DataFrame([summary.__dict__ for summary in summaries])
    if summary_df.empty:
        return summary_df

    summary_df["score"] = normalize(summary_df["score"])
    return summary_df.sort_values("score", ascending=False)


def build_long_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.melt("date", var_name="keyword", value_name="interest").dropna()


def load_stock_csv(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    column_map = {column.lower().strip(): column for column in df.columns}

    date_column = next((column_map[name] for name in ["date", "datetime", "time"] if name in column_map), None)
    close_column = next((column_map[name] for name in ["close", "adj close", "adj_close", "price"] if name in column_map), None)

    if date_column is None or close_column is None:
        raise ValueError("股票 CSV 需要日期欄位 date 與收盤價欄位 close 或 adj close。")

    result = df.rename(columns={date_column: "date", close_column: "close"}).copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result["close"] = pd.to_numeric(result["close"], errors="coerce")

    optional_columns = {}
    for target, names in {
        "open": ["open"],
        "high": ["high"],
        "low": ["low"],
        "volume": ["volume"],
    }.items():
        source = next((column_map[name] for name in names if name in column_map), None)
        if source:
            optional_columns[source] = target

    result = result.rename(columns=optional_columns)
    keep_columns = [column for column in ["date", "open", "high", "low", "close", "volume"] if column in result.columns]
    result = result[keep_columns].dropna(subset=["date", "close"]).sort_values("date")

    if len(result) < 35:
        raise ValueError("資料筆數不足，MACD 與 RSI 回測建議至少 35 筆以上。")

    return result


def add_indicators(
    df: pd.DataFrame,
    fast: int,
    slow: int,
    signal: int,
    rsi_period: int,
) -> pd.DataFrame:
    result = df.copy()
    close = result["close"]

    result["ema_fast"] = close.ewm(span=fast, adjust=False).mean()
    result["ema_slow"] = close.ewm(span=slow, adjust=False).mean()
    result["macd"] = result["ema_fast"] - result["ema_slow"]
    result["macd_signal"] = result["macd"].ewm(span=signal, adjust=False).mean()
    result["macd_hist"] = result["macd"] - result["macd_signal"]

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result["rsi"] = 100 - (100 / (1 + rs))
    result["rsi"] = result["rsi"].fillna(50)

    return result


def backtest_macd_rsi(
    df: pd.DataFrame,
    rsi_buy: int,
    rsi_sell: int,
    initial_cash: float,
    fee_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    result = df.copy()
    macd_cross_up = (result["macd"] > result["macd_signal"]) & (result["macd"].shift(1) <= result["macd_signal"].shift(1))
    macd_cross_down = (result["macd"] < result["macd_signal"]) & (result["macd"].shift(1) >= result["macd_signal"].shift(1))
    result["buy_signal"] = macd_cross_up & (result["rsi"] <= rsi_buy)
    result["sell_signal"] = macd_cross_down | (result["rsi"] >= rsi_sell)

    cash = initial_cash
    shares = 0.0
    position = 0
    trades = []
    equity_values = []

    for _, row in result.iterrows():
        price = float(row["close"])

        if position == 0 and bool(row["buy_signal"]):
            shares = cash * (1 - fee_rate) / price
            cash = 0.0
            position = 1
            trades.append({"date": row["date"], "action": "買進", "price": price, "shares": shares})
        elif position == 1 and bool(row["sell_signal"]):
            cash = shares * price * (1 - fee_rate)
            trades.append({"date": row["date"], "action": "賣出", "price": price, "shares": shares})
            shares = 0.0
            position = 0

        equity_values.append(cash + shares * price)

    result["position"] = 0
    in_position = False
    positions = []
    for _, row in result.iterrows():
        if bool(row["buy_signal"]) and not in_position:
            in_position = True
        elif bool(row["sell_signal"]) and in_position:
            in_position = False
        positions.append(1 if in_position else 0)

    result["position"] = positions
    result["strategy_equity"] = equity_values
    result["buy_hold_equity"] = initial_cash * result["close"] / result["close"].iloc[0]
    result["strategy_return"] = result["strategy_equity"].pct_change().fillna(0)
    result["buy_hold_return"] = result["buy_hold_equity"].pct_change().fillna(0)

    total_return = result["strategy_equity"].iloc[-1] / initial_cash - 1
    buy_hold_return = result["buy_hold_equity"].iloc[-1] / initial_cash - 1
    days = max((result["date"].iloc[-1] - result["date"].iloc[0]).days, 1)
    annual_return = (1 + total_return) ** (365 / days) - 1
    drawdown = result["strategy_equity"] / result["strategy_equity"].cummax() - 1
    max_drawdown = drawdown.min()
    volatility = result["strategy_return"].std(ddof=0) * np.sqrt(252)
    sharpe = annual_return / volatility if volatility else 0

    metrics = {
        "total_return": total_return,
        "buy_hold_return": buy_hold_return,
        "annual_return": annual_return,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "trades": len(trades),
        "final_equity": result["strategy_equity"].iloc[-1],
    }

    return result, pd.DataFrame(trades), metrics


def render_trends_panel() -> None:
    st.subheader("Google Trends 策略熱度")
    uploaded_file = st.file_uploader("上傳 Google Trends CSV", type=["csv"], key="trend_csv")
    lookback = st.slider("策略評估期間", min_value=4, max_value=52, value=12, step=1)

    if uploaded_file is None:
        st.info("請上傳 Google Trends 匯出的 CSV。系統不會自動連線蒐集 Google Trends 數據。")
        st.markdown(
            """
            支援 Google Trends 的 `Interest over time` CSV，或第一欄為 `date`、`day`、`week`、`month` 的一般 CSV。
            """
        )
        return

    try:
        trends_df = load_trends_csv(uploaded_file)
    except Exception as exc:
        st.error(f"CSV 讀取失敗：{exc}")
        return

    summary_df = build_summary(trends_df, lookback)
    long_df = build_long_table(trends_df)

    if summary_df.empty or long_df.empty:
        st.warning("CSV 中沒有足夠的數值資料可供分析。")
        return

    top_keyword = summary_df.iloc[0]
    metric_cols = st.columns(4)
    metric_cols[0].metric("資料起始", str(trends_df["date"].min().date()))
    metric_cols[1].metric("最新日期", str(trends_df["date"].max().date()))
    metric_cols[2].metric("關鍵字數", len(summary_df))
    metric_cols[3].metric("最高優先", top_keyword["keyword"], f"{top_keyword['score']:.1f}")

    fig = px.line(
        long_df,
        x="date",
        y="interest",
        color="keyword",
        markers=True,
        labels={"date": "日期", "interest": "搜尋熱度", "keyword": "關鍵字"},
    )
    fig.update_layout(legend_title_text="", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    display_df = summary_df.rename(
        columns={
            "keyword": "關鍵字",
            "latest": "最新熱度",
            "previous": "前期熱度",
            "average": "期間平均",
            "momentum": "動能",
            "volatility": "波動",
            "score": "策略分數",
            "recommendation": "建議",
        }
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.download_button(
        "下載趨勢策略摘要 CSV",
        data=summary_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="trend_strategy_summary.csv",
        mime="text/csv",
    )


def render_backtest_panel() -> None:
    st.subheader("股票 MACD + RSI 回測")
    stock_file = st.file_uploader("上傳股票價格 CSV", type=["csv"], key="stock_csv")

    settings = st.columns(6)
    fast = settings[0].number_input("MACD 快線", min_value=2, max_value=50, value=12)
    slow = settings[1].number_input("MACD 慢線", min_value=3, max_value=100, value=26)
    signal = settings[2].number_input("MACD 訊號線", min_value=2, max_value=50, value=9)
    rsi_period = settings[3].number_input("RSI 週期", min_value=2, max_value=50, value=14)
    rsi_buy = settings[4].number_input("RSI 買進門檻", min_value=1, max_value=70, value=45)
    rsi_sell = settings[5].number_input("RSI 賣出門檻", min_value=30, max_value=99, value=70)

    capital_cols = st.columns(2)
    initial_cash = capital_cols[0].number_input("初始資金", min_value=1000, value=100000, step=1000)
    fee_rate = capital_cols[1].number_input("單邊交易成本", min_value=0.0, max_value=0.05, value=0.001425, step=0.0001, format="%.4f")

    if stock_file is None:
        st.info("請上傳股票價格 CSV。需要 `date` 與 `close` 欄位，也可包含 open、high、low、volume。")
        st.markdown(
            """
            範例：
            ```csv
            date,open,high,low,close,volume
            2026-01-02,100,103,99,102,1200000
            2026-01-03,102,105,101,104,980000
            ```
            """
        )
        return

    if fast >= slow:
        st.error("MACD 快線週期必須小於慢線週期。")
        return

    try:
        stock_df = load_stock_csv(stock_file)
        indicator_df = add_indicators(stock_df, int(fast), int(slow), int(signal), int(rsi_period))
        backtest_df, trades_df, metrics = backtest_macd_rsi(
            indicator_df,
            int(rsi_buy),
            int(rsi_sell),
            float(initial_cash),
            float(fee_rate),
        )
    except Exception as exc:
        st.error(f"回測失敗：{exc}")
        return

    metric_cols = st.columns(5)
    metric_cols[0].metric("策略總報酬", f"{metrics['total_return']:.2%}")
    metric_cols[1].metric("買進持有", f"{metrics['buy_hold_return']:.2%}")
    metric_cols[2].metric("最大回撤", f"{metrics['max_drawdown']:.2%}")
    metric_cols[3].metric("Sharpe", f"{metrics['sharpe']:.2f}")
    metric_cols[4].metric("交易次數", metrics["trades"])

    price_fig = go.Figure()
    price_fig.add_trace(go.Scatter(x=backtest_df["date"], y=backtest_df["close"], name="Close", mode="lines"))
    buys = backtest_df[backtest_df["buy_signal"]]
    sells = backtest_df[backtest_df["sell_signal"]]
    price_fig.add_trace(go.Scatter(x=buys["date"], y=buys["close"], name="Buy", mode="markers", marker={"symbol": "triangle-up", "size": 11, "color": "#1a9c5b"}))
    price_fig.add_trace(go.Scatter(x=sells["date"], y=sells["close"], name="Sell", mode="markers", marker={"symbol": "triangle-down", "size": 11, "color": "#c23b3b"}))
    price_fig.update_layout(title="價格與交易訊號", hovermode="x unified")
    st.plotly_chart(price_fig, use_container_width=True)

    chart_cols = st.columns(2)
    with chart_cols[0]:
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Scatter(x=backtest_df["date"], y=backtest_df["macd"], name="MACD", mode="lines"))
        macd_fig.add_trace(go.Scatter(x=backtest_df["date"], y=backtest_df["macd_signal"], name="Signal", mode="lines"))
        macd_fig.add_trace(go.Bar(x=backtest_df["date"], y=backtest_df["macd_hist"], name="Histogram"))
        macd_fig.update_layout(title="MACD", hovermode="x unified")
        st.plotly_chart(macd_fig, use_container_width=True)

    with chart_cols[1]:
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(x=backtest_df["date"], y=backtest_df["rsi"], name="RSI", mode="lines"))
        rsi_fig.add_hline(y=rsi_buy, line_dash="dash", line_color="#1a9c5b")
        rsi_fig.add_hline(y=rsi_sell, line_dash="dash", line_color="#c23b3b")
        rsi_fig.update_layout(title="RSI", yaxis_range=[0, 100], hovermode="x unified")
        st.plotly_chart(rsi_fig, use_container_width=True)

    equity_fig = px.line(
        backtest_df,
        x="date",
        y=["strategy_equity", "buy_hold_equity"],
        labels={"date": "日期", "value": "資產淨值", "variable": "策略"},
        title="策略資產淨值 vs 買進持有",
    )
    st.plotly_chart(equity_fig, use_container_width=True)

    st.write("交易紀錄")
    if trades_df.empty:
        st.warning("這組參數沒有觸發交易訊號。")
    else:
        st.dataframe(trades_df, use_container_width=True, hide_index=True)

    st.download_button(
        "下載回測明細 CSV",
        data=backtest_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="macd_rsi_backtest.csv",
        mime="text/csv",
    )


def main() -> None:
    st.set_page_config(page_title="Finance Strategy Trend Analyzer", page_icon="📈", layout="wide")

    st.title("Finance Strategy Trend Analyzer")
    st.caption("手動上傳 CSV，分析 Google Trends 熱度並用 MACD + RSI 進行股票回測。")

    with st.sidebar:
        st.header("資料模式")
        st.success("手動 CSV 上傳")
        st.caption("不自動蒐集 Google Trends 或股票價格資料。")

    trend_tab, backtest_tab = st.tabs(["Google Trends 策略", "股票 MACD + RSI 回測"])

    with trend_tab:
        render_trends_panel()

    with backtest_tab:
        render_backtest_panel()


if __name__ == "__main__":
    main()
