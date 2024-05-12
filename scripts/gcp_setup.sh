#!/bin/bash

# Install docker (https://docs.docker.com/engine/install/ubuntu/)
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

# Install Docker packages
yes | sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin


# Start Docker
sudo service docker start


# Pull docker images
sudo docker pull ${docker_repo}:fetch-ohlcv
sudo docker pull ${docker_repo}:live-trades


# Transfer local files
mkdir -p ~/data/kraken_ohlc_from_api/
mkdir ~/params/
mkdir ~/scripts/
# ex. gcloud compute scp ~/data/kraken_ohlc_from_api/kraken_ohlc_hourly_up2date.csv ${instance-id}:~/data/kraken_ohlc_from_api/kraken_ohlc_hourly_up2date.csv
