load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "rules_python",
    sha256 = "c68bdc4fbec25de5b5493b8819cfc877c4ea299c0dcb15c244c5a00208cde311",
    strip_prefix = "rules_python-0.31.0",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.31.0/rules_python-0.31.0.tar.gz",
)
load("@rules_python//python:repositories.bzl", "py_repositories")
py_repositories()


load("@rules_python//python:pip.bzl", "pip_parse")
pip_parse(
    name = "pip_deps",
    requirements_lock = "//third_party:requirements_lock.txt",
)
load("@pip_deps//:requirements.bzl", "install_deps")
install_deps()


load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "rules_pyvenv",
    strip_prefix = "rules_pyvenv-main",
    url = "https://github.com/cedarai/rules_pyvenv/archive/main.tar.gz",
)