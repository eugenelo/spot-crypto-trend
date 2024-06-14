# Adding Dependencies

To add a pip dependency to the repo, create a new entry in [requirements.txt](../third_party/requirements.txt).

Then, update the lockfile by running the following command:
```
bazel run //third_party:requirements.update
```

The dependency can now be added to a library using the `requirement()` function, e.g. from [ccxt_custom/BUILD.bazel](../ccxt_custom/BUILD.bazel)
```
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
```


When making pull requests, stage & push changes for both [requirements.txt](../third_party/requirements.txt) and the updated [requirements_lock.txt](../third_party/requirements_lock.txt) to the repo.

