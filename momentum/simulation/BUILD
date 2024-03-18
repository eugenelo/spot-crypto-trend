load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "simulation",
    srcs = glob(["src/**/*.py"]),
    imports = ["src"],
    deps = [
        requirement("vectorbt"),
        "//momentum/position_generation:position_generation",
        "//momentum/core:core_nb",
    ],
    visibility = ["//visibility:public"]
)

py_test(
    name="test_backtest",
    srcs = ["test/test_backtest.py"],
    deps = [
        ":simulation"
    ],
    size="small",
)
