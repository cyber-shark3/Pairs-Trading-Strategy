import yfinance as yf
import pandas as pd


def load_data(asset1, asset2, start_date, end_date):
    print(f"Downloading data for {asset1} and {asset2}...")
    data = yf.download(
        [asset1, asset2],
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
    )

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
        if isinstance(close, pd.Series):
            close = close.to_frame(asset1)
        return close[[asset1, asset2]].dropna()

    if "Adj Close" in data.columns:
        return data[["Adj Close"]].rename(columns={"Adj Close": asset1}).dropna()

    return data[["Close"]].rename(columns={"Close": asset1}).dropna()
