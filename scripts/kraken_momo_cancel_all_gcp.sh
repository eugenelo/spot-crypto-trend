#!/bin/bash

gcloud logging write job-status '{ "message": "Canceling all trades"}' --payload-type=json

# Cancel outstanding trades
sudo docker run --rm \
    -v $HOME/kraken_api_key.yaml:$HOME/kraken_api_key.yaml \
    -v $HOME/params/optimize_rohrbach.yaml:$HOME/params/optimize_rohrbach.yaml \
    -v $HOME/data/kraken_ohlc_from_api/kraken_ohlc_hourly_up2date.csv:$HOME/data/kraken_ohlc_from_api/kraken_ohlc_hourly_up2date.csv \
    -v $HOME/logs:/tmp/logs \
    -e USE_STACKDRIVER='true' \
    -w $HOME \
    eugenelo/spot-crypto-trend:live-trades cancel_all --input_path $HOME/data/kraken_ohlc_from_api/kraken_ohlc_hourly_up2date.csv --input_data_freq 1h --output_data_freq 1d --credentials_path $HOME/kraken_api_key.yaml -p $HOME/params/optimize_rohrbach.yaml
cancel_all_status=$?
if [ $cancel_all_status -ne 0 ];
then
    gcloud logging write job-status '{ "message": "cancel-all ran with errors", "success": "false", "step": "cancel-all", "exit_code": "'$cancel_all_status'"}' --payload-type=json --severity=ERROR
    exit 1
fi

# Log success
gcloud logging write job-status '{ "message": "cancel-all finished successfully", "success": "true"}' --payload-type=json

exit 0
