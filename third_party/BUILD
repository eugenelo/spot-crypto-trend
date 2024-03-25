load("@rules_python//python:pip.bzl", "compile_pip_requirements")

# Use like `bazel run //third_party:requirements.update`
compile_pip_requirements(
    name = "requirements",
    requirements_in = "requirements.txt",
    requirements_txt = "requirements_lock.txt",
)