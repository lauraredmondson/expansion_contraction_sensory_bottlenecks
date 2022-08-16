# Expansion and contraction in sensory bottlenecks
Python implementation for 'Expansion and contraction of resource allocation in sensory bottlenecks'. For details, please see [Edmondson, L. R., Jiménez-Rodríguez, A., & Saal, H. P. (2022). Expansion and contraction of resource allocation in sensory bottlenecks. Elife]( https://doi.org/10.7554/eLife.70777).

## Try it online

Click on the badge below to open a fully functional [tutorial notebook](./expansion_contraction_in_sensory_bottlenecks.ipynb) in your browser using myBinder:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lauraredmondson/expansion_contraction_sensory_bottlenecks.git/HEAD?filepath=expansion_contraction_in_sensory_bottlenecks.ipynb)

## Installation
The package requires Python 3.7.6 or higher to run. It also requires *numpy*, *scipy*, *scikit-learn*, and *matplotlib*.

If using conda for package management, the following command creates a new environment *sb* with all dependencies installed:
```conda env create -f environment.yml```

To install a static version of the package, use `python setup.py install`. To install the package in development mode (such that updates to the source directory are reflected in the installed package), use `python setup.py develop`.
