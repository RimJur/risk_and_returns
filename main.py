import yfinance as yf
import polars as pl
import pandas as pd
import numpy as np
import altair as alt


WEEKS_IN_A_YEAR = 52
BRK_B = "BRK-B"
HP = "HP"
TSLA = "TSLA"
GME = "GME"
NVDA = "NVDA"
SPY = "SPY"
TICKERS = [BRK_B, HP, TSLA, GME, NVDA]


def clean_historical_data(
    ticker: str,
    df_pd: pd.DataFrame,
    year_column: str = "year",
    date_column: str = "date",
) -> pl.DataFrame:
    """
    Converts default yfinance pandas dataframe to a polars dataframe with "year", "date", "ticker" and "close" columns
    """

    df_pl = pl.from_pandas(df_pd)
    df_pl = (
        df_pl.cast({"Date": pl.Date})
        .select(pl.col("Date").alias(date_column), pl.col("Close").alias("close"))
        .with_columns(
            year=pl.col("date").dt.year().alias(year_column),
            ticker=pl.lit(ticker),
        )
    )
    return df_pl


def calculate_returns_and_std(df_pl: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate annualized returns and standard deviation from weekly price data.

    Args:
        df_pl: Polars DataFrame with 'close' prices and 'year' column.
        Expected to contain weekly data.
    """
    df_pl = df_pl.with_columns(
        log_returns=np.log(pl.col("close") / pl.col("close").shift(1))
    )

    ticker = df_pl.select(pl.col("ticker"))[0]

    yearly = (
        df_pl.group_by("year")
        .agg(
            yearly_log_returns=pl.col("log_returns").mean(),
            yearly_std=pl.col("log_returns").std(),
            count=pl.count("log_returns"),
        )
        .with_columns(ticker=pl.lit(ticker))
    )

    annualized = (
        yearly.with_columns(
            # Sqrt weeks, because we use std, which is also a sqrt of variance
            annualized_std=pl.col("yearly_std") * np.sqrt(WEEKS_IN_A_YEAR),
            # For log returns, we need to use exp to convert back to simple returns
            annualized_log_returns=(np.exp(pl.col("yearly_log_returns") * 52) - 1),
        )
        .select(
            pl.col("ticker"),
            pl.col("year"),
            pl.col("annualized_log_returns"),
            pl.col("annualized_std"),
        )
        .sort(pl.col("ticker"), pl.col("year"))
    )

    return annualized


def concat_selected_tickers(tickers: list[str]) -> pl.DataFrame:
    ticker_dfs = []
    for ticker in tickers:
        ticker_df = (
            yf.Ticker(ticker).history(period="max", interval="1wk").reset_index()
        )
        ticker_df = clean_historical_data(ticker, ticker_df)
        ticker_df = calculate_returns_and_std(ticker_df)
        ticker_dfs.append(ticker_df)
    return pl.concat(ticker_dfs).filter(pl.col("year") != 2025)


def save_returns_chart(df: pl.DataFrame, file_name: str) -> None:
    base = alt.Chart(df).encode(alt.Color("ticker").legend(None))

    line = base.mark_line(interpolate="monotone").encode(
        x="year:O",
        y="annualized_log_returns",
    )

    last_value = (
        base.mark_circle()
        .encode(
            alt.X("last_date['year']:O"), alt.Y("last_date['annualized_log_returns']:Q")
        )
        .transform_aggregate(
            last_date="argmax(year)",
            groupby=["ticker"],
        )
    )

    company_name = last_value.mark_text(align="left", dx=10, font="monospace").encode(
        text="ticker"
    )

    chart = (
        (line + last_value + company_name)
        .encode(
            x=alt.X("year:O", axis=alt.Axis(labelAngle=-45)).title("Year"),
            y=alt.Y().title("Mean Log Returns"),
        )
        .properties(width=800, title="Annual Mean Returns of 5 Selected Stocks")
    )

    chart.save(file_name)


def save_std_chart(df: pl.DataFrame, file_name: str) -> None:
    base = alt.Chart(df).encode(alt.Color("ticker").legend(None))

    line = base.mark_line(interpolate="monotone").encode(
        x="year:O",
        y="annualized_std",
    )

    last_value = (
        base.mark_circle()
        .encode(alt.X("last_date['year']:O"), alt.Y("last_date['annualized_std']:Q"))
        .transform_aggregate(
            last_date="argmax(year)",
            groupby=["ticker"],
        )
    )

    company_name = last_value.mark_text(align="left", dx=10, font="monospace").encode(
        text="ticker"
    )

    chart = (
        (line + last_value + company_name)
        .encode(
            x=alt.X("year:O", axis=alt.Axis(labelAngle=-45)).title("Year"),
            y=alt.Y().title("Mean Standard Deviation"),
        )
        .properties(
            width=800, title="Annual Mean Standard Deviation of 5 Selected Stocks"
        )
    )

    chart.save(file_name)


def construct_index(combined_df: pl.DataFrame) -> pl.DataFrame:
    df = (
        (
            combined_df.group_by("year")
            .agg(
                pl.col("annualized_log_returns").mean(),
                pl.col("annualized_std").mean(),
            )
            .with_columns(ticker=pl.lit("INDEX"))
        )
        .select(
            pl.col("ticker"),
            pl.col("year"),
            pl.col("annualized_log_returns"),
            pl.col("annualized_std"),
        )
        .sort("year")
    )
    return df


combined_df = concat_selected_tickers(TICKERS)
index_df = construct_index(combined_df)
# save_returns_chart(combined_df, "returns_chart.html")
# save_returns_chart(
#     pl.concat([index_df, concat_selected_tickers([TSLA, SPY])]),
#     "tsla_returns_comparison.html",
# )
# save_returns_chart(
#     pl.concat([index_df, concat_selected_tickers([HP, SPY])]),
#     "hp_returns_comparison.html",
# )
# save_returns_chart(
#     pl.concat([index_df, concat_selected_tickers([GME, SPY])]),
#     "gme_returns_comparison.html",
# )
# save_returns_chart(
#     pl.concat([index_df, concat_selected_tickers([NVDA, SPY])]),
#     "nvda_returns_comparison.html",
# )
# save_returns_chart(
#     pl.concat([index_df, concat_selected_tickers([BRK_B, SPY])]),
#     "brkb_returns_comparison.html",
# )
# save_std_chart(combined_df, "std_chart.html")
# save_std_chart(
#     pl.concat([index_df, concat_selected_tickers([TSLA, SPY])]),
#     "tsla_std_comparison.html",
# )
# save_std_chart(
#     pl.concat([index_df, concat_selected_tickers([HP, SPY])]), "hp_std_comparison.html"
# )
# save_std_chart(
#     pl.concat([index_df, concat_selected_tickers([GME, SPY])]),
#     "gme_std_comparison.html",
# )
# save_std_chart(
#     pl.concat([index_df, concat_selected_tickers([NVDA, SPY])]),
#     "nvda_std_comparison.html",
# )
# save_std_chart(
#     pl.concat([index_df, concat_selected_tickers([BRK_B, SPY])]),
#     "brkb_std_comparison.html",
# )
