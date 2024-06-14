# Building Code

[Bazel](https://bazel.build/) is used to build and test code in the repository. Reference the [user manual](https://bazel.build/docs/user-manual) to familiarize yourself with commands such as `bazel build`, `bazel run`, and `bazel test`.


## Adding Dependencies

After adding a dependency, some additional commands are required to make this dependency reachable in your project's `BUILD` files. See [Adding Dependencies](./adding-dependencies.md) for more info.


## Unit Tests

A useful command for running all user-written tests in the repository is:
```
bazel test -- //... -//third_party/...
```

To disable caching, add `--cache_test_results=no` to the command:
```
bazel test --cache_test_results=no -- //... -//third_party/...
```
