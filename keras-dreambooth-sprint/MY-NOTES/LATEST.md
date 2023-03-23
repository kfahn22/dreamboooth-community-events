# Documentation of things that I have tried


### 3/22/23

Note added Hugging Face token to this environment
`conda create --name Coreml-StableDiffusion-env python=3.8`
`conda activate Coreml-StableDiffusion-env`
`conda install ipykernel`

Install requirements
`pip install huggingface_hub` 
`pip install torch`
`pip install coremltools`
`conda install -c conda-forge diffusers`
`pip install diffusers["torch"] transformers scipy`
`pip install git+https://github.com/apple/ml-stable-diffusion`

Try to install transformers and scipy in notebook

### 3/21/23

`conda create -n stablediffusion-env2 python=3.8`
`conda activate stablediffusion-env2`
`conda install pip`
`conda install ipykernel`
`conda install -c conda-forge tensorflow=2.11.0`


### 3/20/23
`conda create --name StableDiffusion-env python=3.8`
`conda activate StableDiffusion-env`
`conda install pip`
`conda install ipykernel"`
`conda install pytorch torchvision torchaudio -c pytorch-nightly`

## Steps taken on 3/17/23
`conda create -n stablediffusion2 python=3.8`
`conda activate stablediffusion2`
`conda install pytorch torchvision torchaudio -c pytorch-nightly`
`conda install -c conda-forge pandas jupyter jupyterlab`
`conda install -c conda-forge tensorflow=2.11.0`
`conda install -c conda-forge huggingface_hub`

- Run in notebook
`pip install tensorflow-macos===2.11.0 tensorflow-metal===0.7.0 keras_cv===0.4.2 tensorflow_datasets===4.8.1 pillow===9.4.0 imutils opencv-python matplotlib pycocotools`

## Steps Taken (3/14/22)
`conda create -n stablediffusion python=3.8`
`conda activate stablediffusion`
`conda install pytorch torchvision torchaudio -c pytorch-nightly`
`conda install -c conda-forge jupyter jupyterlab`
`conda install -c conda-forge tensorflow=2.11.0`
`conda install -c conda-forge huggingface_hub`
`conda install -c conda-forge -y pandas` (probably should have installed with jupyter)
`pip install --upgrade diffusers[torch]`

- Run in notebook

`pip install tensorflow-macos===2.11.0 tensorflow-metal===0.7.0 keras_cv===0.4.2 tensorflow_datasets===4.8.1 pillow===9.4.0 imutils opencv-python matplotlib pycocotools`

- Issue with sci.py and @rpath/liblapack.3.dylib

`pip install --upgrade --force-reinstall scikit-learn`
[stackoverflow](https://stackoverflow.com/questions/73479899/sklearn-cant-find-lapack-in-new-conda-environment)
[](https://github.com/numpy/numpy/issues/12970)

## Try this suggestion from @AnimeGuru in Discord
conda create --name DreamboothM1 python=3.8
conda activate DreamboothM1
conda install -c conda-forge tensorflow=2.11.0
pip install tensorflow-macos===2.11.0 tensorflow-metal===0.7.0 keras_cv===0.4.2 tensorflow_datasets===4.8.1 pillow===9.4.0 imutils opencv-python matplotlib pycocotools
conda install -c conda-forge -y pandas jupyter

In the notebook setting use_mp = False & tf.keras.mixed_precision.set_global_policy(None)
Also changing the optimizer