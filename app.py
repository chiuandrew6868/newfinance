import io
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


@dataclass
class BacktestMetrics:
    total_return: float
    buy_hold_return: float
    annual_return: float
    max_drawdown: float
    sharpe: float
    trades: int
    final_equity: float


def find_header_row(raw_csv: bytes) -> int:
    preview = raw_csv.decode("utf-8-sig", errors="ignore").splitlines()
    for index, line in enumerate(preview[:30]):
        first_cell = line.split(",", 1)[0].strip().lower()
        if first_cell in {"date", "day", "week", "month", "time"}:
            return index
    return 0


def load_trends_csv(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    df = pd.read_csv(io.BytesIO(raw), skiprows=find_header_row(raw))
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")

    if df.empty or len(df.columns) < 2:
        raise ValueError("CSV 至少需要一個日期欄位與一個趨勢數值欄位。")

    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None).astype("datetime64[ns]")
    df = df.dropna(subset=["date"]).sort_values("date")

    for column in [column for column in df.columns if column != "date"]:
        cleaned = (
            df[column]
            .astype(str)
            .str.replace("<1", "0.5", regex=False)
            .str.replace(",", "", regex=False)
            .str.extract(r"([-+]?\d*\.?\d+)", expand=False)
        )
        df[column] = pd.to_numeric(cleaned, errors="coerce")

    df = df.dropna(axis=1, how="all")
    if len(df.columns) < 2:
        raise ValueError("找不到可用的 Google Trends 數值欄位。")
    return df


@st.cache_data(ttl=60 * 30)
def fetch_stock_data(ticker: str, start: date, end: date) -> pd.DataFrame:
    data = yf.download(
        ticker,
        start=start,
        end=end + timedelta(days=1),
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if data.empty:
        raise ValueError("Yahoo Finance 沒有回傳資料，請確認股票代碼與日期區間。")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()
    data.columns = [str(column).lower().replace(" ", "_") for column in data.columns]
    data = data.rename(columns={"adj_close": "close"})

    keep_columns = [column for column in ["date", "open", "high", "low", "close", "volume"] if column in data]
    data = data[keep_columns].dropna(subset=["date", "close"])
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.tz_localize(None).astype("datetime64[ns]")
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data = data.dropna(subset=["date", "close"]).sort_values("date")

    if len(data) < 60:
        raise ValueError("資料筆數不足，請拉長回測期間。")
    return data


def add_indicators(
    df: pd.DataFrame,
    fast_ema: int,
    slow_ema: int,
    signal_ema: int,
    rsi_period: int,
    rsi_entry_floor: int,
    rsi_overheat_exit: int,
    trend_df: pd.DataFrame | None = None,
    trend_column: str | None = None,
) -> pd.DataFrame:
    result = df.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.tz_localize(None).astype("datetime64[ns]")
    close = result["close"]

    ema_fast = close.ewm(span=fast_ema, adjust=False).mean()
    ema_slow = close.ewm(span=slow_ema, adjust=False).mean()
    result["macd"] = ema_fast - ema_slow
    result["macd_signal"] = result["macd"].ewm(span=signal_ema, adjust=False).mean()
    result["macd_hist"] = result["macd"] - result["macd_signal"]
    result["macd_bullish"] = result["macd"] > result["macd_signal"]

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / rsi_period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / rsi_period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    result["rsi"] = (100 - (100 / (1 + rs))).fillna(50)
    result["rsi_bullish"] = (result["rsi"] >= rsi_entry_floor) & (result["rsi"] < rsi_overheat_exit)

    if trend_df is not None and trend_column:
        trend = trend_df[["date", trend_column]].rename(columns={trend_column: "trend_interest"}).dropna()
        trend["date"] = pd.to_datetime(trend["date"], errors="coerce").dt.tz_localize(None).astype("datetime64[ns]")
        trend["trend_ma"] = trend["trend_interest"].rolling(4, min_periods=1).mean()
        merged = pd.merge_asof(
            result.sort_values("date"),
            trend.sort_values("date"),
            on="date",
            direction="backward",
        )
        result["trend_interest"] = merged["trend_interest"]
        result["trend_ma"] = merged["trend_ma"]
        result["trend_bullish"] = merged["trend_interest"] >= merged["trend_ma"]
    else:
        result["trend_interest"] = np.nan
        result["trend_ma"] = np.nan
        result["trend_bullish"] = False

    return result


def build_combined_signal(df: pd.DataFrame, use_trend: bool, use_macd: bool, use_rsi: bool) -> pd.Series:
    signals = []
    if use_trend:
        signals.append(df["trend_bullish"].fillna(False))
    if use_macd:
        signals.append(df["macd_bullish"].fillna(False))
    if use_rsi:
        signals.append(df["rsi_bullish"].fillna(False))

    if not signals:
        return pd.Series(False, index=df.index)

    combined = signals[0].copy()
    for signal in signals[1:]:
        combined = combined & signal
    return combined


def run_backtest(
    df: pd.DataFrame,
    use_trend: bool,
    use_macd: bool,
    use_rsi: bool,
    initial_cash: float,
    fee_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame, BacktestMetrics]:
    result = df.copy()
    result["signal"] = build_combined_signal(result, use_trend, use_macd, use_rsi)
    result["position"] = result["signal"].shift(1).fillna(False).astype(int)
    result["daily_return"] = result["close"].pct_change().fillna(0)

    trades = []
    previous_position = 0
    for _, row in result.iterrows():
        position = int(row["position"])
        if position != previous_position:
            trades.append(
                {
                    "date": row["date"],
                    "action": "Buy" if position == 1 else "Sell",
                    "price": row["close"],
                }
            )
        previous_position = position

    trade_cost = result["position"].diff().abs().fillna(result["position"]) * fee_rate
    result["strategy_return"] = result["position"] * result["daily_return"] - trade_cost
    result["strategy_equity"] = initial_cash * (1 + result["strategy_return"]).cumprod()
    result["buy_hold_equity"] = initial_cash * (1 + result["daily_return"]).cumprod()

    total_return = result["strategy_equity"].iloc[-1] / initial_cash - 1
    buy_hold_return = result["buy_hold_equity"].iloc[-1] / initial_cash - 1
    days = max((result["date"].iloc[-1] - result["date"].iloc[0]).days, 1)
    annual_return = (1 + total_return) ** (365 / days) - 1
    max_drawdown = (result["strategy_equity"] / result["strategy_equity"].cummax() - 1).min()
    volatility = result["strategy_return"].std(ddof=0) * np.sqrt(252)
    sharpe = annual_return / volatility if volatility else 0

    metrics = BacktestMetrics(
        total_return=float(total_return),
        buy_hold_return=float(buy_hold_return),
        annual_return=float(annual_return),
        max_drawdown=float(max_drawdown),
        sharpe=float(sharpe),
        trades=len(trades),
        final_equity=float(result["strategy_equity"].iloc[-1]),
    )
    return result, pd.DataFrame(trades), metrics


def format_pct(value: float) -> str:
    return f"{value:.2%}"


def main() -> None:
    st.set_page_config(page_title="Finance Strategy Backtester", page_icon="📈", layout="wide")
    st.title("Finance Strategy Backtester")
    st.caption("輸入股票代碼後從 Yahoo Finance 取價，並可勾選 Google Trends、MACD、RSI 進行單頁回測。")

    with st.sidebar:
        st.header("回測設定")
        ticker = st.text_input("股票代碼", value="AAPL", placeholder="例如 AAPL、TSLA、2330.TW").strip().upper()
        start_date = st.date_input("開始時間", value=date.today() - timedelta(days=365 * 3))
        end_date = st.date_input("結束時間", value=date.today())
        initial_cash = st.number_input("初始資金", min_value=1000, value=100000, step=1000)
        fee_rate = st.number_input("單次換倉成本", min_value=0.0, max_value=0.05, value=0.001, step=0.0005, format="%.4f")

        st.divider()
        st.header("技術指標")
        use_trend = st.checkbox("Google Trends", value=True)
        use_macd = st.checkbox("MACD", value=True)
        use_rsi = st.checkbox("RSI", value=True)

        st.divider()
        st.header("指標參數")
        fast_ema = st.number_input("快線 EMA", min_value=2, max_value=100, value=12, step=1)
        slow_ema = st.number_input("慢線 EMA", min_value=3, max_value=250, value=26, step=1)
        signal_ema = st.number_input("訊號線 EMA", min_value=2, max_value=100, value=9, step=1)
        rsi_period = st.number_input("RSI 週期", min_value=2, max_value=100, value=14, step=1)
        rsi_entry_floor = st.number_input("RSI 進場下限", min_value=1, max_value=99, value=50, step=1)
        rsi_overheat_exit = st.number_input("RSI 過熱出場", min_value=1, max_value=100, value=75, step=1)

    trend_df = None
    trend_column = None
    if use_trend:
        uploaded_file = st.file_uploader("上傳 Google Trends CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                trend_df = load_trends_csv(uploaded_file)
                trend_columns = [column for column in trend_df.columns if column != "date"]
                trend_column = st.selectbox("選擇要納入回測的 Trends 欄位", trend_columns)
            except Exception as exc:
                st.error(f"Google Trends CSV 讀取失敗：{exc}")
                st.stop()
        else:
            st.info("你勾選了 Google Trends，請上傳 CSV；若只想用 MACD/RSI，取消勾選 Google Trends。")
            st.stop()

    if not any([use_trend, use_macd, use_rsi]):
        st.warning("請至少勾選一個指標。")
        st.stop()

    if not ticker:
        st.warning("請輸入股票代碼。")
        st.stop()

    if start_date >= end_date:
        st.error("開始日期必須早於結束日期。")
        st.stop()

    if fast_ema >= slow_ema:
        st.error("快線 EMA 必須小於慢線 EMA。")
        st.stop()

    if rsi_entry_floor >= rsi_overheat_exit:
        st.error("RSI 進場下限必須小於 RSI 過熱出場。")
        st.stop()

    try:
        stock_df = fetch_stock_data(ticker, start_date, end_date)
        indicator_df = add_indicators(
            stock_df,
            int(fast_ema),
            int(slow_ema),
            int(signal_ema),
            int(rsi_period),
            int(rsi_entry_floor),
            int(rsi_overheat_exit),
            trend_df,
            trend_column,
        )
        backtest_df, trades_df, metrics = run_backtest(
            indicator_df,
            use_trend,
            use_macd,
            use_rsi,
            float(initial_cash),
            float(fee_rate),
        )
    except Exception as exc:
        st.error(f"回測失敗：{exc}")
        st.stop()

    metric_cols = st.columns(6)
    metric_cols[0].metric("策略報酬", format_pct(metrics.total_return))
    metric_cols[1].metric("買進持有", format_pct(metrics.buy_hold_return))
    metric_cols[2].metric("年化報酬", format_pct(metrics.annual_return))
    metric_cols[3].metric("最大回撤", format_pct(metrics.max_drawdown))
    metric_cols[4].metric("Sharpe", f"{metrics.sharpe:.2f}")
    metric_cols[5].metric("交易次數", metrics.trades)

    st.subheader(f"{ticker} 價格與交易訊號")
    price_fig = go.Figure()
    price_fig.add_trace(go.Scatter(x=backtest_df["date"], y=backtest_df["close"], name="Close", mode="lines"))
    buys = trades_df[trades_df["action"] == "Buy"] if not trades_df.empty else pd.DataFrame()
    sells = trades_df[trades_df["action"] == "Sell"] if not trades_df.empty else pd.DataFrame()
    if not buys.empty:
        price_fig.add_trace(go.Scatter(x=buys["date"], y=buys["price"], name="Buy", mode="markers", marker={"symbol": "triangle-up", "size": 12, "color": "#168f5a"}))
    if not sells.empty:
        price_fig.add_trace(go.Scatter(x=sells["date"], y=sells["price"], name="Sell", mode="markers", marker={"symbol": "triangle-down", "size": 12, "color": "#c2413d"}))
    price_fig.update_layout(hovermode="x unified")
    st.plotly_chart(price_fig, use_container_width=True)

    st.subheader("資金曲線")
    equity_fig = go.Figure()
    equity_fig.add_trace(go.Scatter(x=backtest_df["date"], y=backtest_df["strategy_equity"], name="Strategy", mode="lines"))
    equity_fig.add_trace(go.Scatter(x=backtest_df["date"], y=backtest_df["buy_hold_equity"], name="Buy & Hold", mode="lines"))
    equity_fig.update_layout(hovermode="x unified")
    st.plotly_chart(equity_fig, use_container_width=True)

    chart_cols = st.columns(3)
    with chart_cols[0]:
        st.write("MACD")
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Scatter(x=backtest_df["date"], y=backtest_df["macd"], name="MACD", mode="lines"))
        macd_fig.add_trace(go.Scatter(x=backtest_df["date"], y=backtest_df["macd_signal"], name="Signal", mode="lines"))
        macd_fig.add_trace(go.Bar(x=backtest_df["date"], y=backtest_df["macd_hist"], name="Hist"))
        macd_fig.update_layout(height=320, hovermode="x unified")
        st.plotly_chart(macd_fig, use_container_width=True)

    with chart_cols[1]:
        st.write("RSI")
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(x=backtest_df["date"], y=backtest_df["rsi"], name="RSI", mode="lines"))
        rsi_fig.add_hline(y=rsi_entry_floor, line_dash="dash", line_color="#777")
        rsi_fig.add_hline(y=rsi_overheat_exit, line_dash="dash", line_color="#c2413d")
        rsi_fig.update_layout(height=320, yaxis_range=[0, 100], hovermode="x unified")
        st.plotly_chart(rsi_fig, use_container_width=True)

    with chart_cols[2]:
        st.write("Google Trends")
        trend_fig = go.Figure()
        trend_fig.add_trace(go.Scatter(x=backtest_df["date"], y=backtest_df["trend_interest"], name="Trend", mode="lines"))
        trend_fig.add_trace(go.Scatter(x=backtest_df["date"], y=backtest_df["trend_ma"], name="Trend MA", mode="lines"))
        trend_fig.update_layout(height=320, hovermode="x unified")
        st.plotly_chart(trend_fig, use_container_width=True)

    detail_cols = st.columns(2)
    with detail_cols[0]:
        st.subheader("交易紀錄")
        if trades_df.empty:
            st.info("此期間沒有產生交易。")
        else:
            st.dataframe(trades_df, use_container_width=True, hide_index=True)

    with detail_cols[1]:
        st.subheader("資料預覽")
        preview_columns = ["date", "close", "signal", "position", "macd", "macd_signal", "rsi", "trend_interest"]
        st.dataframe(backtest_df[preview_columns].tail(30), use_container_width=True, hide_index=True)

    st.download_button(
        "下載回測結果 CSV",
        data=backtest_df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{ticker}_backtest.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
