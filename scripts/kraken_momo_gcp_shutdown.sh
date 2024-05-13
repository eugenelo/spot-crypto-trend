#!/bin/bash

user="elo"
home="/home/${user}"

gcloud logging write instance-status '{ "message": "Stopping instance"}' --payload-type=json

gsutil cp gs://kraken_momo/scripts/kraken_momo_cancel_all_gcp.sh ${home}/scripts/kraken_momo_cancel_all_gcp.sh
chmod +x ${home}/scripts/kraken_momo_cancel_all_gcp.sh
su -c ${home}/scripts/kraken_momo_cancel_all_gcp.sh ${user} &> ${home}/logs/shutdown_logs.txt
