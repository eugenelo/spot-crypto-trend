load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "position_generation",
    srcs = glob(["src/**/*.py"]),
    imports = ["src"],
    deps = [
        requirement("vectorbt"),
    ],
    visibility = ["//visibility:public"],
)
