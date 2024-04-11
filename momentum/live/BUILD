load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "constants",
    srcs = [
        "src/live/constants.py",
    ],
    imports = ["src"],
    deps = [],
    visibility = ["//visibility:public"]
)

py_binary(
    name="kraken",
    srcs=[
        "src/live/kraken.py"
    ],
    main="kraken.py",
    deps = [
        requirement("ccxt"),
        requirement("pytz"),
        "//momentum/position_generation:position_generation",
        "//momentum/signal_generation:signal_generation",
        "//momentum/simulation:simulation",
        "//momentum/core:core",
        "//data:data",
        ":constants",
    ],
)
