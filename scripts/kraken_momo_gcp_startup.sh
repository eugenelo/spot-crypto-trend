#!/bin/bash

user="elo"
home="/home/${user}"

gcloud logging write instance-status '{ "message": "Starting instance"}' --payload-type=json


# Register a systemd service to stop worker containers gracefully on shutdown.
gsutil cp gs://kraken_momo/scripts/kraken_momo_gcp_shutdown.sh /etc/systemd/system/kraken_momo_gcp_shutdown.sh
chmod 755 /etc/systemd/system/kraken_momo_gcp_shutdown.sh

# This service will cause the kraken_momo_gcp_shutdown.sh to be invoked before stopping
# docker, hence before tearing down any other container.
cat > /etc/systemd/system/kraken_momo_gcp_shutdown.service <<EOF
[Unit]
Description=Worker container lifecycle
Wants=gcr-online.target docker.service
After=gcr-online.target docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStop=/etc/systemd/system/kraken_momo_gcp_shutdown.sh
EOF

systemctl daemon-reload
systemctl start kraken_momo_gcp_shutdown.service


# Pull latest docker images
sudo docker pull eugenelo/spot-crypto-trend:fetch-ohlcv
sudo docker pull eugenelo/spot-crypto-trend:live-trades


# Run kraken momo workflow
gsutil cp gs://kraken_momo/scripts/run_kraken_momo_gcp.sh ${home}/scripts/run_kraken_momo_gcp.sh
chmod +x ${home}/scripts/run_kraken_momo_gcp.sh
su -c ${home}/scripts/run_kraken_momo_gcp.sh ${user} &> ${home}/logs/startup_logs.txt
