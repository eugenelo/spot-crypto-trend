# Jupyter Notebook

To start a Jupyter Notebook server with access to all of the workspace's dependencies, run the following command:
```
bazel run //notebook:notebook -- --notebook-dir=$(pwd)
```

Imports such as the following can then be made
```
from core.constants import *
```

An example notebook is available at [Crypto Momentum.ipynb](<../../momentum/Crypto Momentum.ipynb>).


## Creating New Packages

To access a newly created package from a notebook, the package must first be added as a dependency to the [`notebook` binary](../../notebook/BUILD.bazel#L5). The server should then be killed and restarted using the same Bazel target as above.
