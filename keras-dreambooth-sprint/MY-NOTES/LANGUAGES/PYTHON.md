## Version Control

- Note:  Torch does not support python version 3.10 
[version control](https://packaging.python.org/en/latest/tutorials/installing-packages/#installing-from-pypi)
`-m pip install "SomeProject>=1,<2"`

## python -m run this module
[discussion of -m](https://stackoverflow.com/questions/50821312/meaning-of-python-m-flag)

Your "default" version is 3.8. It's the first one appearing in your path. Therefore, when you type python3 (Linux or Mac) or python (Windows) in a shell you will start a 3.8 interpreter because that's the first Python executable that is found when traversing your path.

Suppose you are then starting a new project where you want to use Python 3.9. You create a virtual environment called .venv and activate it.

python3.9 -m venv .venv         # "py -3.9" on Windows
source .venv/bin/activate    # ".venv\Scripts\activate" on Windows 
We now have the virtual environment activated using Python 3.9. Typing python in a shell starts the 3.9 interpreter.

BUT, if you type

pip install <some-package>
Then what version of pip is used? Is it the pip for the default version, i.e. Python 3.8, or the Python version within the virtual environment?

An easy way to get around that ambiguity is simply to use

python -m pip install <some-package>
The -m flag makes sure that you are using the pip that's tied to the active Python 

## Notebooks

[Real Python](https://realpython.com/run-python-scripts/)

## Loading scripts

[loading .py file into jupyter nb](https://discourse.jupyter.org/t/how-to-work-with-pure-python-file-py/4443/3)
`import <name of script>` Note you don’t need the .py

[ipython](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-run)
`%run <name of script.py>`


### Import python files into a notebook

`from cases import datelib_spec`
`import importlib`
`importlib.reload(datelib_spec)` # every run

## Errors

python -m python_coreml_stable_diffusion.pipeline --prompt "a photo of an astronaut riding a horse on mars" -i models/coreml-stable-diffusion-v1-4_original_packages -o models/models--apple--coreml-stable-diffusion-v1-4 --compute-unit ALL --seed 93

[path](https://stackoverflow.com/questions/31435921/difference-between-and/55342466#55342466)
[configuration_utils](https://github.com/huggingface/diffusers/blob/main/src/diffusers/configuration_utils.py)
File "/Users/kathrynfahnline/miniconda3/envs/Coreml-StableDiffusion-env/lib/python3.8/site-packages/diffusers/configuration_utils.py", line 371, in load_config
    raise EnvironmentError(
OSError: Can't load config for 'CompVis/stable-diffusion-v1-4'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure 'CompVis/stable-diffusion-v1-4' is the correct path to a directory containing a model_index.json file

Tangentially, don't confuse the directory name . with the Bourne shell command which comprises a single dot (also known by its Bash alias source). The command


`. ./scriptname`


### Issue with version of tensorflow??
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/gpu/tensorflow-0.10.0-py2-none-any.whl

### module not found error
[issues](https://github.com/carson-katri/dream-textures/issues/430)
[multiproccessor?](https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods)

