load("@pip_deps//:requirements.bzl", "requirement")

py_binary(
    name="combine_ohlc",
    srcs = [
        "src/data/combine_ohlc.py",
    ],
    deps = [
        requirement("pandas"),
        requirement("polars"),
    ],
)

py_binary(
    name="tick_to_ohlc",
    srcs = [
        "src/data/tick_to_ohlc.py",
    ],
    deps = [
        requirement("pandas"),
        requirement("polars"),
    ],
)
