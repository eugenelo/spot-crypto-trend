#!/bin/bash

gcloud logging write job-status '{ "message": "Kicking off kraken momo job"}' --payload-type=json

# Fetch new data
sudo docker run --rm \
    -v $HOME/data/kraken_ohlc_from_api/kraken_ohlc_hourly_up2date.csv:$HOME/data/kraken_ohlc_from_api/kraken_ohlc_hourly_up2date.csv \
    -v $HOME/logs:/tmp/logs \
    -e USE_STACKDRIVER='true' \
    -w $HOME \
    genelo33/elo-private:fetch-ohlcv --output_path $HOME/data/kraken_ohlc_from_api/kraken_ohlc_hourly_up2date.csv --data_frequency 1h --lookback_days 30 --append
fetch_data_status=$?
if [ $fetch_data_status -ne 0 ];
then
    gcloud logging write job-status '{ "message": "fetch-ohlcv ran with errors", "success": "false", "step": "fetch-ohlcv", "exit_code": "'$fetch_data_status'"}' --payload-type=json --severity=ERROR
else
    gcloud logging write job-status '{ "message": "fetch-ohlcv ran successfully!", "success": "true", "step": "fetch-ohlcv", "exit_code": "'$fetch_data_status'"}' --payload-type=json
fi

# Cancel outstanding trades
sudo docker run --rm \
    -v $HOME/kraken_api_key.yaml:$HOME/kraken_api_key.yaml \
    -v $HOME/params/optimize_rohrbach.yaml:$HOME/params/optimize_rohrbach.yaml \
    -v $HOME/data/kraken_ohlc_from_api/kraken_ohlc_hourly_up2date.csv:$HOME/data/kraken_ohlc_from_api/kraken_ohlc_hourly_up2date.csv \
    -v $HOME/logs:/tmp/logs \
    -e USE_STACKDRIVER='true' \
    -w $HOME \
    genelo33/elo-private:live-trades cancel_all --input_path $HOME/data/kraken_ohlc_from_api/kraken_ohlc_hourly_up2date.csv --input_data_freq 1h --output_data_freq 1d --credentials_path $HOME/kraken_api_key.yaml -p $HOME/params/optimize_rohrbach.yaml
cancel_all_status=$?
if [ $cancel_all_status -ne 0 ];
then
    gcloud logging write job-status '{ "message": "cancel-all ran with errors", "success": "false", "step": "cancel-all", "exit_code": "'$cancel_all_status'"}' --payload-type=json --severity=ERROR
else
    gcloud logging write job-status '{ "message": "cancel-all ran successfully!", "success": "true", "step": "cancel-all", "exit_code": "'$cancel_all_status'"}' --payload-type=json
fi

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
    -e USE_STACKDRIVER='true' \
    -w $HOME \
    genelo33/elo-private:live-trades execute --input_path $HOME/data/kraken_ohlc_from_api/kraken_ohlc_hourly_up2date.csv --input_data_freq 1h --output_data_freq 1d --credentials_path $HOME/kraken_api_key.yaml -p $HOME/params/optimize_rohrbach.yaml --execution_strategy limit --timezone latest --skip_confirm
execute_trades_status=$?
if [ $execute_trades_status -ne 0 ]
then
    gcloud logging write job-status '{ "message": "execute-trades ran with errors", "success": "false", "step": "execute-trades", "exit_code": "'$execute_trades_status'"}' --payload-type=json --severity=ERROR
    exit 1
else
    gcloud logging write job-status '{ "message": "execute-trades ran successfully!", "success": "true", "step": "execute-trades", "exit_code": "'$execute_trades_status'"}' --payload-type=json
fi

# Log success
gcloud logging write job-status '{ "message": "Job finished successfully", "success": "true"}' --payload-type=json

# Shut down instance
sudo shutdown -h now
