load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "ccxt_custom",
    srcs = glob(["src/ccxt_custom/*.py"]),
    imports = ["src"],
    deps = [
        requirement("ccxt"),
    ],
    visibility = ["//visibility:public"]
)
