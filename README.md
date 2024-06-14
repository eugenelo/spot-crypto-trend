# Spot Cryptocurrency Trend-Following System

This is a trend-following system for spot cryptocurrencies. The pipeline includes:

- Fetching of spot tick & OHLCV data from [Kraken](https://www.kraken.com/)
- Trend signal & position generation
- Position sizing
- Simulation & Backtesting
- Live order management & execution on Kraken

Refer to [Introduction](docs/introduction.md) for a more detailed overview.

***DISCLAIMER: This software is designed primarily for educational purposes and is not intended for professional or commercial trading. It does not provide financial, investment, or trading advice. Use of this software is at your own risk, and the developers are not responsible for any financial losses or damages resulting from its use. Ensure compliance with applicable laws and consult with a licensed financial professional before engaging in any trading activities.***


## Getting Started

### Clone the Repository

This package is not hosted on pip. The easiest way to access the code is via git.

```
# Clone the repository
$ git clone git@github.com:eugenelo/spot-crypto-trend.git

# Change directories into the project
$ cd spot-crypto-trend
```

**NOTE: Each terminal command going forward will be run within the main project directory (i.e. the Bazel workspace directory).**

### Install Requirements

#### 0. Dependencies

This repository requires Python 3.11. The repo and its dependences are managed using [Bazel](https://bazel.build/). Direct dependencies are listed at [third_party/requirements.txt](third_party/requirements.txt) and the full list is autogenerated by pip to [third_party/requirements_lock.txt](third_party/requirements_lock.txt).


#### 1. Install TA-Lib
The backtesting engine used in this project ([vectorbt](https://github.com/polakowo/vectorbt)) depends on [ta-lib-python](https://github.com/ta-lib/ta-lib-python), and thus TA-Lib must be installed. You can find installation instructions [here](https://github.com/TA-Lib/ta-lib-python#dependencies).


#### 2. Install Bazel
It is recommended to install Bazel via [Bazelisk](https://github.com/bazelbuild/bazelisk?tab=readme-ov-file#installation).

Additionally, functions and Bazel rules from [aspect-build/bazel-lib](https://github.com/aspect-build/bazel-lib) and [aspect-build/rules_py](https://github.com/aspect-build/rules_py) are used to enable the building of [OCI](https://github.com/bazel-contrib/rules_oci) (Docker) images. These are already configured in the [`MODULE.bazel`](MODULE.bazel) file and each project's `BUILD.bazel` file, so nothing more needs to be done.


#### 3. Install Remaining Requirements / Build the Code

*Optional: If you are familiar with virtual environments, create one now and activate it. If not, this step is not necessary:*
```
$ python3.11 -m venv .venv
$ source .venv/bin/activate
```

Once both TA-Lib and Bazel have been successfully installed, installation of the remaining requirements can proceed. Bazel will install the requirements listed at [third_party/requirements_lock.txt](third_party/requirements_lock.txt) as it builds the repository.

```
# Build the repository
$ bazel build //...
```

### Environment Setup

#### Generate Kraken API Key

A Kraken Spot API Key will be needed in order to live trade (an API Key is **not needed** for fetching data). Generate a key following the instructions listed [here](https://support.kraken.com/hc/en-us/articles/360000919966-How-to-create-an-API-key#1).

Then, save the credentials to a YAML file in a secure place for later use. The file contents should resemble the following:

```
apiKey: YOUR_API_KEY_HERE
secret: YOUR_API_SECRET_HERE
```


## Running the Code

Various binaries exist within this repo to do things like fetch & manage data, perform data analysis, run backtests, live trade, and more. Refer to [features](docs/features) for more information.

### Live Trading on Local Machine

The following commands can be used to run the main live trading pipeline locally:

```
# Define path to (existing/new) OHLCV data and Kraken API Key
$ export ohlcv_data_path=YOUR_DATA_PATH_HERE
$ export kraken_api_key_path=YOUR_API_KEY_PATH_HERE

# Fetch latest hourly OHLCV data. Update existing if available, else create new file.
$ bazel run //data:fetch_kraken_ohlcv_data -- --output_path ${ohlcv_data_path} --data_frequency 1h --lookback_days 30 --append

# Cancel any outstanding trades
$ bazel run //live:trades -- cancel_all --input_path ${ohlcv_data_path} --input_data_freq 1h --output_data_freq 1d --credentials_path ${kraken_api_key_path} -p momentum/params/optimize_rohrbach.yaml

# Update positions and execute trades
$ bazel run //live:trades -- execute --input_path ${ohlcv_data_path} --input_data_freq 1h --output_data_freq 1d --credentials_path ${kraken_api_key_path} -p momentum/params/optimize_rohrbach.yaml --timezone latest --execution_strategy market
```

Refer to [Live Trading](docs/features/live-trading.md) for more details.


### Docker

Docker images exist for the live trading pipeline and can be used to run the pipeline **without needing to clone the repo or build any code**. The same basic steps from above (fetch hourly OHLCV data, cancel outstanding trades, execute new trades) are run. Refer to [run_kraken_momo_docker.sh](scripts/run_kraken_momo_docker.sh) for an example.

Pull the image tags from:
```
docker pull eugenelo/spot-crypto-trend:fetch-ohlcv
docker pull eugenelo/spot-crypto-trend:live-trades
```

**NOTE: Only the most recent 30 days of hourly OHLCV data can be fetched from the [Kraken OHLC REST API](https://docs.kraken.com/rest/#tag/Spot-Market-Data/operation/getOHLCData) due to inbuilt limitations. If running the pipeline through Docker, it is assumed that you have existing hourly OHLCV data for all assets in your universe up till at most 30 days from today. Refer to [Converting Tick to OHLCV Data](docs/features/converting-tick-to-ohlcv-data.md) for instructions on creating this data from tick data if necessary.**


### Google Cloud Platform

The live trading pipeline can be set up to execute automatically on a [Google Cloud Compute (GCP)](https://cloud.google.com/products/compute) instance at a regular interval (e.g. daily). Integrations with [Cloud Logging](https://cloud.google.com/logging) have been made to enable alerts for job statuses, job completion, and errors. Refer to [Running on GCP](docs/running-on-gcp.md) for more details.
