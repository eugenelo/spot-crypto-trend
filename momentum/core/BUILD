load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "core",
    srcs = [
        "src/core/constants.py",
        "src/core/utils.py",
    ],
    imports = ["src"],
    deps = [
        requirement("pandas"),
        requirement("numpy"),
    ],
    visibility = ["//visibility:public"]
)

py_library(
    name = "core_nb",
    srcs = [
        "src/core/utils_nb.py",
    ],
    imports = ["src"],
    deps = [
        requirement("numba"),
    ],
    visibility = ["//visibility:public"]
)