import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from housing_rates_compare import (
    load_zillow_city_zhvi_by_city,
    load_fed_funds,
    TARGET_CITIES,
    TARGET_STATE,
)

st.set_page_config(page_title="OC Housing vs Fed Rate", layout="wide")

@st.cache_data(show_spinner=False)
def get_price_city() -> pd.DataFrame:
    df = load_zillow_city_zhvi_by_city()
    df["Month"] = df["Month"].dt.to_period("M").dt.to_timestamp()
    return df

@st.cache_data(show_spinner=False)
def get_fed() -> pd.DataFrame:
    df = load_fed_funds()
    df["Month"] = df["Month"].dt.to_period("M").dt.to_timestamp()
    return df

@st.cache_data(show_spinner=False)
def get_joined() -> pd.DataFrame:
    prices = get_price_city()
    fed = get_fed()
    joined = prices.merge(fed, on="Month", how="inner")
    return joined


def filter_date(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df[(df["Month"] >= start) & (df["Month"] <= end)].copy()


def compute_average(prices_city: pd.DataFrame, cities: list[str]) -> pd.DataFrame:
    sub = prices_city[prices_city["RegionName"].isin(cities)].copy()
    avg = (
        sub.groupby("Month")["ZHVI_YoY_pct"].mean().rename("Avg_YoY_pct").reset_index()
    )
    return avg


def make_chart(df: pd.DataFrame, cities: list[str], show_avg: bool, show_fed: bool) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Colors
    colors = {
        "Anaheim": "#1f77b4",
        "Garden Grove": "#2ca02c",
        "Buena Park": "#9467bd",
        "Average": "#111111",
    }

    # Per-city traces
    for city in cities:
        grp = df[df["RegionName"] == city].sort_values("Month")
        if grp.empty:
            continue
        fig.add_trace(
            go.Scatter(x=grp["Month"], y=grp["ZHVI_YoY_pct"], mode="lines", name=f"{city} YoY%", line=dict(color=colors.get(city))),
            secondary_y=False,
        )

    # Average trace
    if show_avg and cities:
        avg = compute_average(df[["Month", "RegionName", "ZHVI_YoY_pct"]].drop_duplicates(), cities)
        fig.add_trace(
            go.Scatter(x=avg["Month"], y=avg["Avg_YoY_pct"], mode="lines", name="Average YoY%", line=dict(color=colors["Average"], width=3, dash="dash")),
            secondary_y=False,
        )

    # Fed funds
    if show_fed:
        fed = df.loc[:, ["Month", "FedFunds"]].drop_duplicates(subset=["Month"]).sort_values("Month")
        fig.add_trace(
            go.Scatter(x=fed["Month"], y=fed["FedFunds"], mode="lines", name="Federal Funds Rate (%)", line=dict(color="#d62728")),
            secondary_y=True,
        )

    # 0% reference line on primary y-axis
    try:
        fig.add_hline(y=0, line_width=1, line_color="#888888")
    except Exception:
        # Fallback for older Plotly versions: add as a shape tied to primary y-axis
        x_min = df["Month"].min()
        x_max = df["Month"].max()
        fig.add_shape(type="line", x0=x_min, x1=x_max, y0=0, y1=0, xref="x", yref="y", line=dict(color="#888888", width=1))

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=40, t=60, b=40),
        height=600,
    )
    fig.update_xaxes(title_text="Month")
    fig.update_yaxes(title_text="House Price YoY %", secondary_y=False)
    fig.update_yaxes(title_text="Federal Funds Rate (%)", secondary_y=True)
    return fig


def correlation_table(df: pd.DataFrame, cities: list[str], show_avg: bool) -> pd.DataFrame:
    out_rows = []
    fed = df.loc[:, ["Month", "FedFunds"]].drop_duplicates(subset=["Month"]).set_index("Month")
    for city in cities:
        grp = df[df["RegionName"] == city].loc[:, ["Month", "ZHVI_YoY_pct"]].dropna()
        merged = grp.merge(fed, on="Month", how="inner")
        if not merged.empty:
            corr = merged[["ZHVI_YoY_pct", "FedFunds"]].corr().iloc[0, 1]
            out_rows.append({"Series": f"{city} YoY%", "Correlation vs FedFunds": corr})
    if show_avg and cities:
        avg = compute_average(df[["Month", "RegionName", "ZHVI_YoY_pct"]], cities)
        merged = avg.merge(fed.reset_index(), on="Month", how="inner").dropna()
        if not merged.empty:
            corr = merged[["Avg_YoY_pct", "FedFunds"]].corr().iloc[0, 1]
            out_rows.append({"Series": "Average YoY%", "Correlation vs FedFunds": corr})
    return pd.DataFrame(out_rows)


# Sidebar controls
st.sidebar.header("Filters")
joined = get_joined()
min_month = joined["Month"].min()
max_month = joined["Month"].max()

cities_sel = st.sidebar.multiselect(
    "Cities", options=TARGET_CITIES, default=TARGET_CITIES
)

date_range = st.sidebar.slider(
    "Date range",
    min_value=min_month.to_pydatetime(),
    max_value=max_month.to_pydatetime(),
    value=(max(min_month.to_pydatetime(), max_month.to_pydatetime().replace(year=max_month.year-10)), max_month.to_pydatetime()),
    format="YYYY-MM",
)

show_avg = st.sidebar.checkbox("Show average of selected cities", value=True, label_visibility="collapsed")
show_fed = st.sidebar.checkbox("Show Federal Funds Rate", value=True, label_visibility="collapsed")

# Filter
start, end = [pd.to_datetime(d).to_period("M").to_timestamp() for d in date_range]
subset = filter_date(joined, start, end)
subset = subset[subset["RegionName"].isin(cities_sel)]

st.title("Orange County Housing YoY% vs Federal Funds Rate")
st.caption(f"Cities: {', '.join(cities_sel)} | State: {TARGET_STATE}")

# Handle insufficient data for YoY (needs 12 months history)
if subset["ZHVI_YoY_pct"].dropna().empty:
    st.warning("Selected date range may be too short for YoY calculations. Please extend the range.")
else:
    fig = make_chart(subset, cities_sel, show_avg, show_fed)
    # Streamlit deprecation: use width instead of use_container_width
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)

    corr_df = correlation_table(subset, cities_sel, show_avg)
    if not corr_df.empty:
        st.subheader("Correlation vs Federal Funds (within selected range)")
        st.dataframe(corr_df.style.format({"Correlation vs FedFunds": "{:.3f}"}), use_container_width=True)

st.markdown("---")
st.caption("ZHVI: Zillow Home Value Index for mid-tier single-family homes (seasonally adjusted, monthly). Data: Zillow Research, FRED.")
