# Running on GCP

This document describes how to automatically run the live trading software on [Google Cloud Compute (GCP)](https://cloud.google.com/products/compute) with daily rebalancing.

## Create GCP instance

The first step is to create a GCP VM instance. For those with no prior experience, Google provides [helpful tutorials](https://cloud.google.com/compute/docs/create-linux-vm-instance).

The [Docker images](./building-docker-images.md) which we will be using are built off of the base [`@ubuntu` Docker image](https://hub.docker.com/_/ubuntu), and so we should create an instance which uses the Ubuntu operating system. `Ubuntu 20.04 LTS` is a known working version.

Around 1GB of RAM is required to run the program. We will be loading the ~600MB OHLCV CSV into a dataframe twice (once to update the data, and again to generate positions). Some of the shared-core instances (`e2-small` or `e2-medium`) are technically sufficiently capable, but loading the dataframe could take an extremely long time (10+ minutes) if preempted. A non-preemptible machine type of `e2-standard-2` or better is recommended.

A 10GB persistent disk is sufficient for everything we need to upload to the instance (Docker images ~2GB and hourly OHLCV data ~0.6GB), but leaves us with little wiggle room. In our experience, resolving issues with a full persistent disk can be a headache. If the disk is sufficiently full, one can no longer SSH into the instance to clear space. [Resizing a disk](https://cloud.google.com/compute/docs/disks/resize-persistent-disk) can often fail due to insufficient resource availability. We suggest allocating more space to the disk than needed as a safety buffer.


## Set up instance

SSH into the instance after it has been created.

We will need to set up Docker. First, **follow the [installation instructions for Ubuntu](https://docs.docker.com/engine/install/ubuntu/).**

Then, start the service
```
$ sudo service docker start
```

Pull the images of interest
```
$ docker pull eugenelo/spot-crypto-trend:fetch-ohlcv
$ docker pull eugenelo/spot-crypto-trend:live-trades
```

We also need to transfer some files onto the instance, specifically:

- Hourly OHLCV CSV file
- Kraken API Key
- Position Generation Parameters (YAML file)

The [default script](../scripts/run_kraken_momo_gcp.sh) assume the following locations for each file:

- OHLCV CSV: `~/data/kraken_ohlc_from_api/kraken_ohlc_hourly_up2date.csv`
- Kraken API Key: `~/kraken_api_key.yaml`
- Position Generation Params: `~/params/optimize_rohrbach.yaml`

**SCP these files from your local machine to the instance.**

These steps are contained in the setup script [gcp_setup.sh](../scripts/gcp_setup.sh).


## Upload scripts to cloud storage

A few scripts are fetched from predefined locations by default. [Create a bucket](https://cloud.google.com/storage/docs/creating-buckets) named `kraken_momo` and upload the following from the repo root directory
```
$ gsutil cp scripts/kraken_momo_gcp_startup.sh gs://kraken_momo/scripts/kraken_momo_gcp_startup.sh
$ gsutil cp scripts/kraken_momo_gcp_shutdown.sh gs://kraken_momo/scripts/kraken_momo_gcp_shutdown.sh
$ gsutil cp scripts/run_kraken_momo_gcp.sh gs://kraken_momo/scripts/run_kraken_momo_gcp.sh
$ gsutil cp scripts/kraken_momo_cancel_all_gcp.sh gs://kraken_momo/scripts/kraken_momo_cancel_all_gcp.sh
```


## Assign startup script to instance

To kick off the live trading job automatically, we will utilize startup scripts. The startup script of interest is [kraken_momo_gcp_startup.sh](../scripts/kraken_momo_gcp_startup.sh). The script will:

- Register a systemd service for graceful shutdown (using [kraken_momo_gcp_shutdown.sh](../scripts/kraken_momo_gcp_shutdown.sh))
- Pull the latest docker images
- Pull the latest version of [run_kraken_momo_gcp.sh](../scripts/run_kraken_momo_gcp.sh)
- Run `run_kraken_momo_gcp.sh` as `$user` (refer to [section below](#what-does-the-live-trading-job-actually-consist-of) for details)

This script runs as root on startup, hence the switch user command. `$user` should be changed to the username under which the local files were transferred during [setup](#set-up-instance).

There are a number of ways to [add a startup script to your VM](https://cloud.google.com/compute/docs/instances/startup-scripts/linux). Our recommendation is to store the script in a [Cloud Storage Bucket](https://cloud.google.com/storage?hl=en) to be passed to the VM via the instance metadata. This provides a convenient way to modify the contents of the startup script without having to make further changes to the instance settings. For example, if your script was uploaded to `gs://kraken_momo/scripts/kraken_momo_gcp_startup.sh` you would add the key-value pair
```
startup-script-url: gs://kraken_momo/scripts/kraken_momo_gcp_startup.sh
```
to the instance metadata.

### Why no shutdown script?

GCP also supports specifying [shutdown scripts](https://cloud.google.com/compute/docs/shutdownscript) in a very similar manner to startup scripts. Why do we register a systemd service for graceful shutdown rather than making use of this feature?

Graceful shutdown consists of [cancelling all open trades](../scripts/kraken_momo_cancel_all_gcp.sh) before terminating the instance. This is done through the `live-trades` image and thus requires the docker service to be alive. It turns out that GCP shutdown scripts are run as root

GCP shutdown scripts are also executed on a best-effort basis and are subject to [limitations](https://cloud.google.com/compute/docs/shutdownscript#limitations) such as a maximum running time. Canceling open trades should be fairly quick, but the lack of execution guarantees are still undesirable. Registering the shutdown behavior as a systemd service gets around these limitations.


## Create and assign instance schedule

The final step is to automate the startup and shutdown of the instance. A couple options exist for this: [instance schedules](https://cloud.google.com/compute/docs/instances/schedule-instance-start-stop) and the [Cloud Scheduler](https://cloud.google.com/scheduler/docs/start-and-stop-compute-engine-instances-on-a-schedule). The live trading job is entirely compatible with instance schedules, and we recommend using these over the Cloud Scheduler for simplicity.

Due to how [OHLCV data is fetched and stored](./features/fetching-kraken-ohlcv-data.md#dropping-incomplete-rows), we recommend starting the job after the turn of the hour. This allows the maximum amount of fetched data to be used during position generation. Liquidity / trading volume on Kraken can fluctuate but we generally find that it correlates with US stock market hours (increased volume on trading days, decreased volume overnight and on weekends / holidays). For simplicity, we recommend configuring a schedule which spins up an instance around market open hours (09:30 ET) every day.

The live trading job is already set to [shut the instance down on completion](#what-does-the-live-trading-job-actually-consist-of). This decreases unnecessary uptime of our instance and reduces cost. But in case of errors, we can configure the schedule to guarantee that the instance is always shut down after a certain time as a failsafe.


## Set up logging alerts

[Alerting policies](https://cloud.google.com/logging/docs/alerting/log-based-alerts) can be created for automatic monitoring of job statuses and errors. Examples of useful policies and the associated log query may include:

- Live Trading Job Started
  ```
  resource.type="global" log_name="projects/{project}/logs/instance-status" jsonPayload.message="Starting instance"
  ```
- Live Trading Job Completed Successfully
  ```
  resource.type="global" log_name="projects/{project}/logs/job-status" jsonPayload.message="Job finished successfully"
  ```
- Live Trading Job Exited with Error
  ```
  severity=ERROR resource.type="global" log_name="projects/{project}/logs/job-status"
  ```
- Instance was Shutdown
  ```
  resource.type="gce_instance" protoPayload.methodName="v1.compute.instances.stop" operation.last="true"
  ```


## What does the live trading job actually consist of?

The live trading job is executed from [run_kraken_momo_gcp.sh](../scripts/run_kraken_momo_gcp.sh). The job consists of the following main steps:

1. [Fetch the latest hourly OHLCV data](./features/fetching-kraken-ohlcv-data.md) while dropping incomplete rows
2. Cancel all outstanding trades
3. [Run the live trading loop](./features/live-trading.md)
4. Shut down the instance after the live trading loop has completed

If either (1) or (2) fail, the script will abort without running (3) and log an error.
