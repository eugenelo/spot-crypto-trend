load("@pip_deps//:requirements.bzl", "requirement")

py_binary(
    name="kraken",
    srcs = [
        "src/kraken/kraken.py",
    ],
    deps = [
        requirement("ccxt"),
        requirement("pytz"),
        "//momentum/position_generation:position_generation",
        "//momentum/signal_generation:signal_generation",
        "//momentum/simulation:simulation",
        "//momentum/core:core",
        "//data:data",
    ],
)

py_binary(
    name="convert_kraken_symbols",
    srcs = [
        "src/kraken/convert_kraken_symbols.py",
    ],
    deps = [
        requirement("ccxt"),
        requirement("pytz"),
        "//data:data",
    ],
)
