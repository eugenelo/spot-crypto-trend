load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "data",
    srcs = [
        "src/data/constants.py",
        "src/data/utils.py",
    ],
    imports = ["src"],
    deps = [
        requirement("pandas"),
        requirement("pytz"),
    ],
    visibility = ["//visibility:public"]
)

py_binary(
    name="combine_ohlc",
    srcs = [
        "src/data/combine_ohlc.py",
    ],
    deps = [
        requirement("pandas"),
        requirement("polars"),
        ":data",
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
