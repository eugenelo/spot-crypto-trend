load("@pip_deps//:requirements.bzl", "requirement")

py_binary(
    name="momentum",
    srcs=["main.py"],
    main="main.py",
    deps = [
        requirement("PyYAML"),
        "//momentum/analysis:analysis",
        "//momentum/core:core",
        "//momentum/position_generation:position_generation",
        "//momentum/signal_generation:signal_generation",
        "//momentum/simulation:simulation",
        "//data:data",
    ],
)
