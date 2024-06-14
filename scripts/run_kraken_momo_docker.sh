#!/bin/bash

# Fetch new data
sudo docker run --rm \
    -v $HOME/data/kraken_ohlc_from_api/kraken_ohlc_hourly_up2date.csv:$HOME/data/kraken_ohlc_from_api/kraken_ohlc_hourly_up2date.csv \
    -v $HOME/logs:/tmp/logs \
    -e USE_STACKDRIVER='false' \
    -w $HOME \
    eugenelo/spot-crypto-trend:fetch-ohlcv --output_path $HOME/data/kraken_ohlc_from_api/kraken_ohlc_hourly_up2date.csv --data_frequency 1h --lookback_days 30 --append
fetch_data_status=$?

# Cancel outstanding trades
sudo docker run --rm \
    -v $HOME/kraken_api_key.yaml:$HOME/kraken_api_key.yaml \
    -v $HOME/params/optimize_rohrbach.yaml:$HOME/params/optimize_rohrbach.yaml \
    -v $HOME/data/kraken_ohlc_from_api/kraken_ohlc_hourly_up2date.csv:$HOME/data/kraken_ohlc_from_api/kraken_ohlc_hourly_up2date.csv \
    -v $HOME/logs:/tmp/logs \
    -e USE_STACKDRIVER='false' \
    -w $HOME \
    eugenelo/spot-crypto-trend:live-trades cancel_all --input_path $HOME/data/kraken_ohlc_from_api/kraken_ohlc_hourly_up2date.csv --input_data_freq 1h --output_data_freq 1d --credentials_path $HOME/kraken_api_key.yaml -p $HOME/params/optimize_rohrbach.yaml
cancel_all_status=$?

# Exit early if steps failed
if [ $fetch_data_status -ne 0 ] || [ $cancel_all_status -ne 0 ];
then
    exit 1
fi

# Execute new trades
sudo docker run --rm \
    -v $HOME/kraken_api_key.yaml:$HOME/kraken_api_key.yaml \
    -v $HOME/params/optimize_rohrbach.yaml:$HOME/params/optimize_rohrbach.yaml \
    -v $HOME/data/kraken_ohlc_from_api/kraken_ohlc_hourly_up2date.csv:$HOME/data/kraken_ohlc_from_api/kraken_ohlc_hourly_up2date.csv \
    -v $HOME/logs:/tmp/logs \
    -e USE_STACKDRIVER='false' \
    -w $HOME \
    eugenelo/spot-crypto-trend:live-trades execute --input_path $HOME/data/kraken_ohlc_from_api/kraken_ohlc_hourly_up2date.csv --input_data_freq 1h --output_data_freq 1d --credentials_path $HOME/kraken_api_key.yaml -p $HOME/params/optimize_rohrbach.yaml --execution_strategy limit-then-market --timezone latest --skip_confirm
execute_trades_status=$?
exit $execute_trades_status
