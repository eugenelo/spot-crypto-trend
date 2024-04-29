load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "live",
    srcs = [
        "src/live/constants.py",
        "src/live/utils.py",
        "src/live/execution.py",
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
    name="trades",
    srcs=[
        "src/live/trades.py"
    ],
    main="trades.py",
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


py_test(
    name="test_execution",
    srcs = ["test/test_execution.py"],
    deps = [
        ":live",
    ],
    size="medium",
)
