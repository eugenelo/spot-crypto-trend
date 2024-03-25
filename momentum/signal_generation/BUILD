load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "signal_generation",
    srcs = glob(["src/**/*.py"]),
    imports = ["src"],
    deps = [
        requirement("pandas"),
        requirement("plotly"),
        requirement("numpy"),
        requirement("pandas-ta"),
    ],
    visibility = ["//visibility:public"],
)

py_test(
    name="test_common",
    srcs = ["test/test_common.py"],
    deps = [
        ":signal_generation"
    ],
    size="small",
)
