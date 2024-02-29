import requests
import pandas as pd


def fetch_historical_futures_data(symbol, since):
    """
    Fetch historical futures data for a given symbol from Kraken.

    :param symbol: The symbol to fetch historical data for (e.g., 'PI_XBTUSD')
    :param since: UNIX timestamp to fetch historical entries since
    :return: DataFrame with historical data
    """
    # URL for the Kraken Futures historical data API endpoint
    url = f"https://futures.kraken.com/api/history/v3/market/{symbol}/executions"

    # Parameters for the API request
    params = {
        "symbol": symbol,
        "since": since,
    }

    # Make the GET request
    response = requests.get(url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        # Load the JSON data
        data = response.json()
        print(len(data["elements"]))

        # Assuming the API returns data in a format that includes timestamps and prices, convert to DataFrame
        # This is a placeholder and should be adjusted based on the actual structure of the response
        df = pd.DataFrame(
            data["data"],
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )

        # Convert timestamps to readable dates
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

        return df
    else:
        raise Exception("Failed to fetch data from Kraken Futures API")


if __name__ == "__main__":
    # Example usage
    symbol = "PI_XBTUSD"  # Example symbol
    since = "1609459200"  # Example UNIX timestamp (January 1, 2021)
    data = fetch_historical_futures_data(symbol, since)
    print(data)
