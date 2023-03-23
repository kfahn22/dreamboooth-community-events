# NOTES ON CONDA

## Conda
[conda](https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links)
- Setting up environment  
[conda basics](https://gist.github.com/atifraza/b1a92ae7c549dd011590209f188ed2a0)
`conda create --name basic-ml-env python<=3.9 pip`

- Turn off automatic activiation of environment (after installing anaconda)
`conda config --set auto_activate_base false`

- Removing an environment with pip
`cd /path/to/env_name` 
`rm -rf env name`

- Removing an environment with conda
`conda env remove -n env name`

Install in a subdirectory
`conda create --prefix ./env python<=3.9 pip`

## Base environment issue
[base](https://stackoverflow.com/questions/54429210/how-do-i-prevent-conda-from-activating-the-base-environment-by-default)
