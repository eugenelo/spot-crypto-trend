load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "analysis",
    srcs = glob(["src/**/*.py"]),
    imports = ["src"],
    deps = [
        requirement("pandas"),
        requirement("plotly"),
        requirement("scipy"),
        requirement("statsmodels"),
        "//data:data",
    ],
    visibility = ["//visibility:public"]
)
