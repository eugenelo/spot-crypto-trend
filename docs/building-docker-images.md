# Building Docker Images

Targets exist to build and publish Docker image tags for the live trading pipeline binaries.

These images are built off of the base [`@ubuntu` Docker image](https://hub.docker.com/_/ubuntu) and are compatible with both AArch64 and x86_64 architectures.


## Fetch OHLCV Data

```
# Build the tarball
$ bazel run //data:tarball

# (Optional) Run the binary using Docker
$ docker run --rm eugenelo/spot-crypto-trend:fetch-ohlcv

# Publish the image to Docker hub
$ bazel run //data:publish
```


## Execute Live Trades

```
# Build the tarball
$ bazel run //live:tarball

# (Optional) Run the binary using Docker
$ docker run --rm eugenelo/spot-crypto-trend:live-trades

# Publish the image to Docker hub
$ bazel run //live:publish
```


## Pulling Images

To pull the latest published tags,
```
$ docker pull eugenelo/spot-crypto-trend:fetch-ohlcv
$ docker pull eugenelo/spot-crypto-trend:live-trades
```


## Updating the Image Repository

To change the image repository location, change the `repo_tags`, `repository`, and `remote_tags` fields in the oci targets accordingly. E.g. from [data/BUILD.bazel](../data/BUILD.bazel)
```
oci_tarball(
    name = "tarball",
    format = "docker",
    image = ":platform_image",
    repo_tags = ["eugenelo/spot-crypto-trend:fetch-ohlcv"],      <--
)

oci_push(
    name = "publish",
    image = ":platform_image",
    repository = "eugenelo/spot-crypto-trend",                   <--
    remote_tags = ["fetch-ohlcv"],                          <--
)
```
