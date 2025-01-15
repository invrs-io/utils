# invrs-utils - Miscellaneous utilities
![Continuous integration](https://github.com/invrs-io/utils/actions/workflows/build-ci.yml/badge.svg)
![PyPI version](https://img.shields.io/pypi/v/invrs-utils)

This package is a collection of utilities that may be useful, but do not have fundamental roles in the invrs-io ecosystem. These currently include,

- `experiment.checkpoint`: Defines a simple checkpoint manager with an [orbax](https://github.com/google/orbax)-like API.
- `experiment.sweep`: Functions to construct experiments involving hyperparameter sweeps.
- `experiment.data`: Functions to load and summarize data from experiments.
- `experiment.work_unit`: Provides a function to run a single work unit in an experiment.

## Install
```
pip install invrs-utils
```
