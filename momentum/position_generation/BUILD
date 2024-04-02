load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "position_generation",
    srcs = glob(["src/**/*.py"]),
    imports = ["src"],
    deps = [
        "//momentum/signal_generation:signal_generation",
    ],
    visibility = ["//visibility:public"],
)
