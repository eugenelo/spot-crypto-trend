load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "simulation",
    srcs = glob(["src/simulation/*.py"]),
    deps = [
        ":vbt",
        "//momentum/position_generation:position_generation",
        "//momentum/core:core",
    ],
    visibility = ["//visibility:public"]
)

# Use vbtpro
py_library(
    name="vbt",
    srcs = glob(["src/simulation/vbt/*.py"]),
    imports = ["src"],
    deps = [
        ":simulation_vbtpro",
    ],
    visibility = ["//visibility:public"]
)

py_library(
    name="simulation_vbtdev",
    srcs = glob(["src/simulation/vbt/vbtdev/*.py"]),
    imports = ["src"],
    deps = [
        requirement("vectorbt"),
        "//momentum/core:core_nb",
    ],
    visibility = ["//visibility:private"]
)

py_library(
    name="simulation_vbtpro",
    srcs = glob(["src/simulation/vbt/vbtpro/*.py"]),
    imports = ["src"],
    deps = [
        requirement("vectorbtpro"),
        "//momentum/core:core_nb",
    ],
    visibility = ["//visibility:private"]
)

py_test(
    name="test_simulate",
    srcs = ["test/test_simulate.py"],
    deps = [
        ":vbt",
    ],
    size="medium",
)

py_test(
    name="test_utils",
    srcs = ["test/test_utils.py"],
    deps = [
        ":simulation"
    ],
    size="small",
)

py_test(
    name="test_optimize",
    srcs = ["test/test_optimize.py"],
    deps = [
        ":simulation"
    ],
    size="small",
)
