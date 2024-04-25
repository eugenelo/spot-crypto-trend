load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "live",
    srcs = [
        "src/live/constants.py",
        "src/live/utils.py",
    ],
    imports = ["src"],
    deps = [
        requirement("ccxt"),
        "//data:data",
        "//ccxt_custom:ccxt_custom",
    ],
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
        ":live",
    ],
)

py_binary(
    name="pnl",
    srcs=[
        "src/live/pnl.py"
    ],
    main="pnl.py",
    deps = [
        requirement("ccxt"),
        requirement("pytz"),
        "//momentum/position_generation:position_generation",
        "//momentum/signal_generation:signal_generation",
        "//momentum/simulation:simulation",
        "//momentum/core:core",
        "//data:data",
        ":live",
    ],
)
