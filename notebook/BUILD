load("@pip_deps//:requirements.bzl", "requirement")
load("@pip_deps//:requirements.bzl", "all_requirements")

py_binary(
  name = "notebook",
  srcs = ["src/notebook/notebook.py"],
  deps = [
    requirement("ccxt"),
    requirement("pytz"),
    requirement("jupyterlab"),
    requirement("notebook"),
    "//data:data",
    "//momentum/analysis:analysis",
    "//momentum/core:core",
    "//momentum/position_generation:position_generation",
    "//momentum/signal_generation:signal_generation",
    "//momentum/simulation:simulation",
  ],
)
