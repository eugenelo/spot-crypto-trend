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
        requirement("polars"),
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
        requirement("tqdm"),
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
        requirement("tqdm"),
        ":data",
    ],
)

py_binary(
    name="fix_write_stamp_alignment",
    srcs = [
        "src/data/fix_write_stamp_alignment.py",
    ],
    deps = [
        requirement("pandas"),
        requirement("tqdm"),
        ":data",
    ],
)

py_binary(
    name="fetch_kraken_tick_data",
    srcs = [
        "src/data/fetch_kraken_tick_data.py",
    ],
    deps = [
        requirement("ccxt"),
        requirement("pytz"),
        requirement("tqdm"),
        ":data",
    ],
)

py_binary(
    name="fetch_kraken_ohlcv_data",
    srcs = [
        "src/data/fetch_kraken_ohlcv_data.py",
    ],
    deps = [
        requirement("ccxt"),
        requirement("pytz"),
        requirement("tqdm"),
        ":data",
    ],
)

py_binary(
    name="convert_kraken_symbols",
    srcs = [
        "src/data/convert_kraken_symbols.py",
    ],
    deps = [
        requirement("ccxt"),
        requirement("pytz"),
        ":data",
    ],
)

py_test(
    name="test_utils",
    srcs = ["test/test_utils.py"],
    deps = [
        ":data",
    ],
    size="small",
)
