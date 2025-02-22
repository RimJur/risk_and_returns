import yfinance as yf
import polars as pl

BRK_B = "BRK-B"
HP = "HP"
TSLA = "TSLA"
GME = "GME"
NVDA = "NVDA"
TICKERS = [BRK_B, HP, TSLA, GME, NVDA]


for ticker in TICKERS:
    ticker_df = yf.Ticker(ticker).history(period="1y").reset_index()

    df_pl = pl.from_pandas(ticker_df)
    df_pl = df_pl.cast({"Date": pl.Date}).with_columns(
        ticker=pl.lit(ticker),
    )
    df_pl.write_csv(f"extracted_data/{ticker}.csv")
