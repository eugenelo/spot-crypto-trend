load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "signal_generation",
    srcs = glob(["src/**/*.py"]),
    imports = ["src"],
    deps = [
        requirement("pandas"),
        requirement("plotly"),
        requirement("numpy"),
    ],
    visibility = ["//visibility:public"],
)
