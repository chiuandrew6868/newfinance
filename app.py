import io
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
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


def render_empty_state() -> None:
    st.info("請先上傳從 Google Trends 匯出的 CSV。上傳後才會開始分析，系統不會自動連線蒐集資料。")
    st.markdown(
        """
        支援格式：
        - Google Trends 匯出的 `Interest over time` CSV
        - 一般 CSV，第一欄為 `date`、`day`、`week` 或 `month`
        - 後續欄位為關鍵字或主題，數值範圍通常為 0 到 100
        """
    )


def main() -> None:
    st.set_page_config(
        page_title="Finance Strategy Trend Analyzer",
        page_icon="📈",
        layout="wide",
    )

    st.title("Finance Strategy Trend Analyzer")
    st.caption("手動上傳 Google Trends CSV 後分析市場熱度、動能與策略優先級。")

    with st.sidebar:
        st.header("資料來源")
        uploaded_file = st.file_uploader("上傳 Google Trends CSV", type=["csv"])
        lookback = st.slider("策略評估期間", min_value=4, max_value=52, value=12, step=1)
        st.divider()
        st.write("目前模式")
        st.success("手動 CSV 上傳")
        st.caption("已取消自動蒐集 Google Trends 數據。")

    if uploaded_file is None:
        render_empty_state()
        st.stop()

    try:
        trends_df = load_trends_csv(uploaded_file)
    except Exception as exc:
        st.error(f"CSV 讀取失敗：{exc}")
        st.stop()

    summary_df = build_summary(trends_df, lookback)
    long_df = build_long_table(trends_df)

    if summary_df.empty or long_df.empty:
        st.warning("CSV 中沒有足夠的數值資料可供分析。")
        st.stop()

    top_keyword = summary_df.iloc[0]
    latest_date = trends_df["date"].max().date()
    start_date = trends_df["date"].min().date()

    metric_cols = st.columns(4)
    metric_cols[0].metric("資料起始", str(start_date))
    metric_cols[1].metric("最新日期", str(latest_date))
    metric_cols[2].metric("關鍵字數", len(summary_df))
    metric_cols[3].metric("最高優先", top_keyword["keyword"], f"{top_keyword['score']:.1f}")

    st.subheader("趨勢走勢")
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

    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("策略優先級")
        display_df = summary_df.copy()
        display_df = display_df.rename(
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
        st.dataframe(
            display_df.style.format(
                {
                    "最新熱度": "{:.1f}",
                    "前期熱度": "{:.1f}",
                    "期間平均": "{:.1f}",
                    "動能": "{:+.1f}",
                    "波動": "{:.1f}",
                    "策略分數": "{:.1f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    with right:
        st.subheader("策略摘要")
        leaders = summary_df.head(3)
        laggards = summary_df.tail(3).sort_values("score")

        st.write("優先投入")
        for _, row in leaders.iterrows():
            st.markdown(
                f"- **{row['keyword']}**：{row['recommendation']}，"
                f"最新熱度 {row['latest']:.1f}，動能 {row['momentum']:+.1f}。"
            )

        st.write("需要降噪")
        for _, row in laggards.iterrows():
            st.markdown(
                f"- **{row['keyword']}**：{row['recommendation']}，"
                f"波動 {row['volatility']:.1f}，策略分數 {row['score']:.1f}。"
            )

    st.subheader("原始資料預覽")
    st.dataframe(trends_df, use_container_width=True, hide_index=True)

    csv_download = summary_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "下載策略摘要 CSV",
        data=csv_download,
        file_name="trend_strategy_summary.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
