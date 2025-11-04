import os
import io
import sys
import textwrap
from datetime import datetime
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import requests

ZILLOW_ZHVI_CITY_URL = (
    # Mid-tier single family homes, seasonally adjusted, monthly
    "https://files.zillowstatic.com/research/public_csvs/zhvi/City_zhvi_uc_sfr_tier_0.33_0.67_sm_sa_month.csv"
)
FRED_FEDFUNDS_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS"
TARGET_CITIES = ["Anaheim", "Garden Grove", "Buena Park"]
TARGET_STATE = "CA"


def fetch_csv(url: str) -> pd.DataFrame:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text))


def load_zillow_city_zhvi() -> pd.DataFrame:
    df = fetch_csv(ZILLOW_ZHVI_CITY_URL)
    # Expected columns include: RegionName (city), StateName, and monthly date columns like '2000-01'
    # Keep only target cities in CA
    df = df[(df.get("StateName") == TARGET_STATE) & (df.get("RegionName").isin(TARGET_CITIES))]
    if df.empty:
        raise RuntimeError(
            "No ZHVI rows found for target cities. Zillow format may have changed."
        )

    # Identify date columns (YYYY-MM)
    date_cols = [c for c in df.columns if c[:2].isdigit() and ("-" in c or "/" in c)]
    # Melt to long format
    long_df = df.melt(
        id_vars=["RegionName", "StateName"],
        value_vars=date_cols,
        var_name="Month",
        value_name="ZHVI",
    )
    # Standardize date
    # Zillow sometimes uses YYYY-MM or M/D/YYYY; handle both
    def parse_month(val: str) -> pd.Timestamp:
        val = str(val)
        for fmt in ("%Y-%m", "%m/%d/%Y", "%Y-%m-%d"):
            try:
                return pd.to_datetime(val, format=fmt)
            except Exception:
                pass
        # Fallback: let pandas try
        return pd.to_datetime(val, errors="coerce")

    long_df["Month"] = long_df["Month"].map(parse_month)
    long_df = long_df.dropna(subset=["Month", "ZHVI"]).copy()
    long_df["ZHVI"] = pd.to_numeric(long_df["ZHVI"], errors="coerce")
    long_df = long_df.dropna(subset=["ZHVI"])  # drop if not numeric

    # Compute YoY percent change per city
    long_df.sort_values(["RegionName", "Month"], inplace=True)
    long_df["ZHVI_YoY_pct"] = (
        long_df.groupby("RegionName")["ZHVI"].pct_change(12) * 100.0
    )

    # Aggregate across the 3 cities: simple average of YoY
    agg = (
        long_df.dropna(subset=["ZHVI_YoY_pct"])  # need YoY available
        .groupby("Month")["ZHVI_YoY_pct"].mean()
        .rename("Price_YoY_pct")
        .to_frame()
        .reset_index()
    )
    return agg


def load_zillow_city_zhvi_by_city() -> pd.DataFrame:
    """Return YoY percent change per city for target cities.

    Columns: Month, RegionName, ZHVI_YoY_pct
    """
    df = fetch_csv(ZILLOW_ZHVI_CITY_URL)
    df = df[(df.get("StateName") == TARGET_STATE) & (df.get("RegionName").isin(TARGET_CITIES))]
    if df.empty:
        raise RuntimeError("No ZHVI rows found for target cities. Zillow format may have changed.")

    date_cols = [c for c in df.columns if c[:2].isdigit() and ("-" in c or "/" in c)]
    long_df = df.melt(
        id_vars=["RegionName", "StateName"],
        value_vars=date_cols,
        var_name="Month",
        value_name="ZHVI",
    )

    def parse_month(val: str) -> pd.Timestamp:
        val = str(val)
        for fmt in ("%Y-%m", "%m/%d/%Y", "%Y-%m-%d"):
            try:
                return pd.to_datetime(val, format=fmt)
            except Exception:
                pass
        return pd.to_datetime(val, errors="coerce")

    long_df["Month"] = long_df["Month"].map(parse_month)
    long_df = long_df.dropna(subset=["Month", "ZHVI"]).copy()
    long_df["ZHVI"] = pd.to_numeric(long_df["ZHVI"], errors="coerce")
    long_df = long_df.dropna(subset=["ZHVI"])  # drop if not numeric

    long_df.sort_values(["RegionName", "Month"], inplace=True)
    long_df["ZHVI_YoY_pct"] = (
        long_df.groupby("RegionName")["ZHVI"].pct_change(12) * 100.0
    )
    # Keep only YoY rows
    out = long_df.dropna(subset=["ZHVI_YoY_pct"]).loc[:, ["Month", "RegionName", "ZHVI_YoY_pct"]]
    return out


def load_fed_funds() -> pd.DataFrame:
    df = fetch_csv(FRED_FEDFUNDS_URL)
    # Normalize column names to avoid surprises
    norm_cols = {c: c.strip().upper() for c in df.columns}
    df.rename(columns=norm_cols, inplace=True)
    # Columns are typically: DATE or OBSERVATION_DATE, and FEDFUNDS
    rename_map = {}
    if "DATE" in df.columns:
        rename_map["DATE"] = "Month"
    if "OBSERVATION_DATE" in df.columns:
        rename_map["OBSERVATION_DATE"] = "Month"
    if "FEDFUNDS" in df.columns:
        rename_map["FEDFUNDS"] = "FedFunds"
    df.rename(columns=rename_map, inplace=True)

    if "Month" not in df.columns or "FedFunds" not in df.columns:
        raise RuntimeError(
            f"Unexpected FRED CSV columns: {list(df.columns)}. Could not find DATE/FEDFUNDS."
        )

    df["Month"] = pd.to_datetime(df["Month"])  # already month-end dates
    df["FedFunds"] = pd.to_numeric(df["FedFunds"], errors="coerce")
    df = df.dropna(subset=["FedFunds"]).copy()
    return df


def build_joined_series() -> pd.DataFrame:
    prices = load_zillow_city_zhvi()
    fed = load_fed_funds()

    # Align to monthly start; inner join on month
    prices["Month"] = prices["Month"].dt.to_period("M").dt.to_timestamp()
    fed["Month"] = fed["Month"].dt.to_period("M").dt.to_timestamp()

    joined = pd.merge(prices, fed, on="Month", how="inner")
    # Reasonable date window where both exist
    joined = joined[(joined["Month"] >= joined["Month"].min())].copy()
    return joined


def build_joined_series_per_city() -> pd.DataFrame:
    prices_city = load_zillow_city_zhvi_by_city()
    fed = load_fed_funds()

    prices_city["Month"] = prices_city["Month"].dt.to_period("M").dt.to_timestamp()
    fed["Month"] = fed["Month"].dt.to_period("M").dt.to_timestamp()

    joined = pd.merge(prices_city, fed, on="Month", how="inner")
    return joined


def plot_series(df: pd.DataFrame, out_path: str | None = None) -> None:
    plt.style.use("seaborn-v0_8")
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(df["Month"], df["Price_YoY_pct"], color="#1f77b4", label="House Price YoY% (avg of 3 cities)")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("House Price YoY %", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    # 0% reference line
    ax1.axhline(0, color="#888888", linewidth=1)

    ax2 = ax1.twinx()
    ax2.plot(df["Month"], df["FedFunds"], color="#d62728", label="Federal Funds Rate (%)")
    ax2.set_ylabel("Federal Funds Rate (%)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    # Title and legend
    title = (
        "Anaheim, Garden Grove, Buena Park: House Price YoY% vs Federal Funds Rate"
    )
    plt.title(title)

    # Build combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150)
        print(f"Saved plot to {out_path}")
    else:
        plt.show()


def plot_city_series(df: pd.DataFrame, out_path: str | None = None) -> None:
    plt.style.use("seaborn-v0_8")
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot YoY per city
    colors = {
        "Anaheim": "#1f77b4",
        "Garden Grove": "#2ca02c",
        "Buena Park": "#9467bd",
    }
    for city, grp in df.sort_values(["RegionName", "Month"]).groupby("RegionName"):
        ax1.plot(grp["Month"], grp["ZHVI_YoY_pct"], label=f"{city} YoY%", color=colors.get(city))

    ax1.set_xlabel("Month")
    ax1.set_ylabel("House Price YoY %")

    # 0% reference line
    ax1.axhline(0, color="#888888", linewidth=1)

    # Fed funds on secondary axis (single series)
    ax2 = ax1.twinx()
    fed = df.loc[:, ["Month", "FedFunds"]].drop_duplicates(subset=["Month"]).sort_values("Month")
    ax2.plot(fed["Month"], fed["FedFunds"], color="#d62728", label="Federal Funds Rate (%)")
    ax2.set_ylabel("Federal Funds Rate (%)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    title = "House Price YoY% by City vs Federal Funds Rate"
    plt.title(title)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150)
        print(f"Saved plot to {out_path}")
    else:
        plt.show()


def summarize(df: pd.DataFrame) -> None:
    # Quick correlation and basic regression stats (without external deps)
    # Pearson correlation
    corr = df[["Price_YoY_pct", "FedFunds"]].corr().iloc[0, 1]
    print(f"Correlation (YoY price vs Fed Funds): {corr:.3f}")

    # Simple OLS via numpy polyfit
    try:
        import numpy as np
        x = df["FedFunds"].to_numpy()
        y = df["Price_YoY_pct"].to_numpy()
        mask = ~(pd.isna(x) | pd.isna(y))
        b1, b0 = np.polyfit(x[mask], y[mask], deg=1)  # y = b1*x + b0
        print(f"Simple OLS (y = b1*x + b0): b1={b1:.3f}, b0={b0:.3f}")
    except Exception as e:
        print(f"Regression summary skipped: {e}")


def main(out_file: str | None = None):
    print("Fetching and modeling data…")
    df = build_joined_series()
    print(
        textwrap.dedent(
            f"""
            Date range: {df['Month'].min().date()} to {df['Month'].max().date()} ({len(df)} months)
            Cities: {', '.join(TARGET_CITIES)} (state: {TARGET_STATE})
            """
        ).strip()
    )
    summarize(df)

    if out_file is None:
        out_dir = os.path.join(os.getcwd(), "output")
        out_file = os.path.join(out_dir, "housing_vs_fedfunds.png")
    plot_series(df, out_file)

    # Also produce by-city chart
    df_city = build_joined_series_per_city()
    out_dir = os.path.dirname(out_file) if out_file else os.path.join(os.getcwd(), "output")
    out_city = os.path.join(out_dir, "housing_vs_fedfunds_by_city.png")
    plot_city_series(df_city, out_city)


if __name__ == "__main__":
    outfile = None
    if len(sys.argv) > 1:
        outfile = sys.argv[1]
    main(outfile)
