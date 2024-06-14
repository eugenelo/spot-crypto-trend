# Code Style

This repository uses [Black](https://github.com/psf/black) and [Flake8](https://github.com/PyCQA/flake8/tree/main) to lint Python code. Black and [isort](https://pycqa.github.io/isort/) are used for auto-formatting and import sorting.

`make lint` can be used from the main project directory to run the linters.

`make fix` can be used from the main project directory to run the code formatters.

The Flake8 config is located at [.flake8](../.flake8). The isort config is located in [pyproject.toml](../pyproject.toml).
