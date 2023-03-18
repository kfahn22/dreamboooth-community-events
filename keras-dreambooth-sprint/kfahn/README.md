# Trying to change code to work with Torch on m1 mac using mps

## APPLE DOCUMENTATION AND RESOURCES
[Apple research](https://machinelearning.apple.com)
[Resources](https://developer.apple.com/machine-learning/resources/)
[WWWDC22](https://developer.apple.com/videos/all-videos/)
[metal-code](https://developer.apple.com/metal/sample-code/)

### tensorflow-metal
[tensorflow-metal](https://developer.apple.com/metal/tensorflow-plugin/)
`bash ~/miniconda.sh -b -p $HOME/miniconda`  
`source ~/miniconda/bin/activate`
`conda install -c apple tensorflow-deps`

[speed issues](https://github.com/pytorch/pytorch/issues/77799)

### Conda
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

## Installing pytorch on apple m1 
[Apple docs Metal](https://developer.apple.com/metal/pytorch/)

`conda install pytorch torchvision torchaudio -c pytorch`

[pytorch](https://towardsdatascience.com/installing-pytorch-on-apple-m1-chip-with-gpu-acceleration-3351dc44d67c)
`conda create -n stablediffusion python=3.8`
[mps backend](https://pytorch.org/docs/master/notes/mps.html)
[gpu-acceleration](https://medium.com/@angelgaspar/how-to-install-tensorflow-on-a-m1-m2-macbook-with-gpu-acceleration-acfeb988d27e)
[installing pytorch](https://www.youtube.com/watch?v=WqSCr8NezLQ)
[installing pytorch on mac in vsc](https://www.youtube.com/watch?v=WqSCr8NezLQ)
`pip install torch`
`import torch`
`mpsDevice = torch.device("mps" if torch.backends.mps.is_available() else cpu)`

## Moving from tensorflow to pytorch
[Moving](https://neptune.ai/blog/moving-from-tensorflow-to-pytorch)
[arrays](https://www.tutorialspoint.com/how-to-convert-a-numpy-ndarray-to-a-pytorch-tensor-and-vice-versa)
[images to tensors](https://towardsdatascience.com/convert-images-to-tensors-in-pytorch-and-tensorflow-f0ab01383a03)

## Tensorflow and pytorch
[tf and pytorch](https://stackoverflow.com/questions/74704866/running-tf-and-torch-on-one-virtual-environment)
[pytorh vs nightly](https://discuss.pytorch.org/t/pytorch-nightly-vs-stable/105633)
[torch on mps](https://pytorch.org/docs/master/notes/mps.html)

## Pytorch Fundamentals
#----convert numpy to tensor--*
X_tensor = torch.from_numpy(x).float().to(device)
#----convert tensor back to numpy
×_cpu = x_tensor.cpu().numpy() # first convert tensor to cpu (from a gpu tensor)
#--requires_grad = True or False to make a variable trainable or not
w = torch.randn(1, requires_grad=True, type=torch. float).to(device)
b = torch.randn(1, requires_grad=True, type=torch. float). to(device)
w. requires_grad_() # functions that end with .
do inplace modification
b.requires_grad_(False)

# We can specify the device at the moment of creation - RECOMMENDED!
torch.manual_seed(42)
a = torch.randn(1, requires_grad=True, type=torch. float, device=device)
## XLA
[XLA](https://github.com/pytorch/xla)
`pip3 install torch_xla[tpuvm]`

## Pytorch optimization and training
ops
[torch ops](https://github.com/pytorch/pytorch/blob/master/torch/_ops.py)
[pytorch-lightning](https://pytorch-lightning.readthedocs.io/en/stable/accelerators/mps_basic.html)
`PYTORCH_ENABLE_MPS_FALLBACK=1 python your_script.py `

to enable training on GPUS
`trainer = Trainer(accelerator="mps", devices=1)`

## Class on Torch
[Lecture](http://kiwi.bridgeport.edu/cpeg589/CPEG589_Lecture4.pdf)

### Pytorch Issues
[github pytorch](https://github.com/pytorch/pytorch)
Mixed precision??
[github isssue](https://github.com/pytorch/pytorch/issues/88415)
[Bias](https://stackoverflow.com/questions/55229636/import-lstm-from-tensorflow-to-pytorch-by-hand?rq=1)
[AMX?](https://discuss.pytorch.org/t/how-to-check-mps-availability/152015/9)
[seed](https://stackoverflow.com/questions/74614882/manual-seed-for-mps-devices-in-pytorch)
`import torch` 
`torch.manual_seed(0)`

### Converting data to torch
[dataset conversion](https://stackoverflow.com/questions/67345480/converting-a-tf-dataset-to-a-pytorch-dataset)

`import tensorflow datasets as tfds import torch.nn as nn`  
`def train dataloader (batch size):`  
`return tfds.as_numpy(tfds.load('mnist').batch(batch_size))`  
`class Model (n.Module):`  
`def forward(self, x):`  
`× = torch. as_tensor (x, device='mps')`  

## Base environment issue
[base](https://stackoverflow.com/questions/54429210/how-do-i-prevent-conda-from-activating-the-base-environment-by-default)

### Version Control

- Note:  Torch does not support python version 3.10 
[version control](https://packaging.python.org/en/latest/tutorials/installing-packages/#installing-from-pypi)
`-m pip install "SomeProject>=1,<2"`

### Issue with version of tensorflow??
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/gpu/tensorflow-0.10.0-py2-none-any.whl

## Examples
[Moving from keras to pytorch](https://towardsdatascience.com/moving-from-keras-to-pytorch-f0d4fff4ce79)
[Kaggle Notebook](https://www.kaggle.com/code/mlwhiz/third-place-model-for-toxic-comments-in-pytorch/notebook)


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


## Steps taken on 3/17/23
`conda create -n stablediffusion2 python=3.8`
`conda activate stablediffusion2`
`conda install pytorch torchvision torchaudio -c pytorch-nightly`
`conda install -c conda-forge pandas jupyter jupyterlab`
`conda install -c conda-forge tensorflow=2.11.0`
`conda install -c conda-forge huggingface_hub`

- Run in notebook
`pip install tensorflow-macos===2.11.0 tensorflow-metal===0.7.0 keras_cv===0.4.2 tensorflow_datasets===4.8.1 pillow===9.4.0 imutils opencv-python matplotlib pycocotools`


## Loss scale optimizer error
[loss scale optimizer](https://keras.io/api/mixed_precision/loss_scale_optimizer/#baselossscaleoptimizer-class)

[github](https://github.com/keras-team/keras/blob/v2.11.0/keras/mixed_precision/loss_scale_optimizer.py#L361-L586)

[adam](https://github.com/keras-team/keras/blob/v2.11.0/keras/optimizers/legacy/adam.py)


- [pipeline.py](https://github.com/kjsman/stable-diffusion-pytorch/blob/main/stable_diffusion_pytorch/pipeline.py)


## KERAS-CV

### clip_tokenizer.py

[CLIP](https://github.com/openai/CLIP)
-there is an example on README using torch

-clip_tokenizer.py utilizes the openai CLIP bpe_simple_vocab_16e6.txt file?

[clip](https://github.com/openai/CLIP/tree/main/clip)

### diffusion_model.py

-need file_hash
[model weights](https://huggingface.co/ianstenbit/keras-sd2.1/resolve/main/diffusion_model_v2_1.h5)
-using swish activation -- find out about it

### image_encoder.py

- autoencoder
- using AttentionBlock, PaddedConv2D, ResnetBlock
- need file_hash
[encorder weights](https://huggingface.co/fchollet/stable-diffusion/resolve/main/vae_encoder.h5)

### noise_scheduler.py

### text_encoder.py

- need file_hash 
- version 1
[weights]https://huggingface.co/fchollet/stable-diffusion/resolve/main/kcv_encoder.h5()
- version 2
[weight](https://huggingface.co/ianstenbit/keras-sd2.1/resolve/main/text_encoder_v2_1.h5)

## DIFFUSERS

[Apple-ml-stable-diffusion](https://github.com/apple/ml-stable-diffusion)
[stability-ai](https://github.com/Stability-AI/stablediffusion)
[CompVis](https://github.com/CompVis/stable-diffusion)
[hugging-face]( https://huggingface.co/docs/diffusers/quicktour)
[Another repo](https://github.com/kjsman/stable-diffusion-pytorch)
[](https://github.com/TheLastBen/fast-stable-diffusion)


## Subscriptions

[pay.google.com](https://payments.google.com/payments?esp=AJ9oCCws41N1fXKEen47%2BGbs68yZUg3dfd7IYR12jq3O70uxw5MlElLrcSArWx33vkeGmlNU7UoKrKS5z4NWoclBidvSlR%2BCw4f9NrDo9UjEZHRU37F8HTJFPForBWdjllfsNu79%2BTgA&authuser=0)

### Kaggle Example
[Fine-tune-your-own-stable-diffusion](https://www.kaggle.com/code/enricobeltramo/fine-tune-your-own-stable-diffusion[])

## Working on m1 mac
[hugging face-cli](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/login)
[hg-working on m1 mac](https://huggingface.co/docs/diffusers/optimization/mps)

-- module not found error
[issues](https://github.com/carson-katri/dream-textures/issues/430)
[multiproccessor?](https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods)

this is from 12/2021 maybe fixed??
[arm64](https://developer.apple.com/forums/thread/695963)

Requirements: 
- arm64 version of Python
- Pytorch 1.13

[Apple m1 on mac](https://wandb.ai/morgan/stable-diffusion/reports/Running-Stable-Diffusion-on-an-Apple-M1-Mac-With-HuggingFace-Diffusers--VmlldzoyNTU2ODc2)

## Keras
[keras examples](https://keras.io/examples/)
[keras_cv](https://github.com/kfahn22/keras-cv/tree/master/keras_cv/models/stable_diffusion)
[Fine-tuning stable-diffusion](https://keras.io/examples/generative/finetune_stable_diffusion/)

## HUGGING-FACE
[installation](https://huggingface.co/docs/huggingface_hub/installation)

`pip install 'huggingface_hub[cli,torch]'`
[login issues](https://discuss.huggingface.co/t/huggingface-cli-login/25567/2)
[Diffusers discussion](https://discuss.huggingface.co/c/discussion-related-to-httpsgithubcomhuggingfacediffusers/63)

# DREAMBOOTH

## HUGGING FACE keras-dreambooth-sprint
[diffusors](https://github.com/huggingface/diffusers)
[kerasCV](https://huggingface.co/docs/diffusers/using-diffusers/kerascv)
[convert-kerascv-sd-diffusers](https://huggingface.co/spaces/sayakpaul/convert-kerascv-sd-diffusers)
[app.py](https://huggingface.co/spaces/sayakpaul/convert-kerascv-sd-diffusers/blob/main/app.py)

## sprint

[Sprint](https://github.com/huggingface/community-events/blob/main/keras-dreambooth-sprint/Dreambooth_on_Hub.ipynb)
[Lambda](https://github.com/huggingface/community-events/blob/main/keras-dreambooth-sprint/compute-with-lambda.md)
[Leaderboard](https://huggingface.co/keras-dreambooth)

## GitHub repos working on training DreamBooth on Mac
[XavierXiao](https://github.com/XavierXiao/Dreambooth-Stable-Diffusion/pull/36)
[DreanBoothMac](https://github.com/SujeethJinesh/DreamBoothMac)
[Issues](https://github.com/CompVis/stable-diffusion/issues/25)

## backend.py
[error re dtype](https://github.com/keras-team/keras/blob/master/keras/backend.py)

## CORE-ML

[Apple Docs](https://coremltools.readme.io/docs/introductory-quickstart) 
[core-ml](https://developer.apple.com/machine-learning/core-ml/)
[tensorflow-2](https://coremltools.readme.io/docs/tensorflow-2)
`pip install tensorflow==2.2.0 h5py==2.10.0 coremltools pillow`

Steps:
In terminal:
`conda create --name Dreambooth_Coreml python=3.8`  
`conda activate Dreambooth_coreml`  
`conda install pip`
`pip install coremltools pillow`

[h5py](https://docs.h5py.org/en/stable/build.html)


## Hugging Face coreml
[Hugging-Face-conversion](https://huggingface.co/blog/diffusers-coreml)
`pip install huggingface_hub`
`pip install git+https://github.com/apple/ml-stable-diffusion`
[coverted](https://huggingface.co/apple/coreml-stable-diffusion-2-base)
[stable-diffusion-coreml](https://machinelearning.apple.com/research/stable-diffusion-coreml-apple-silicon) 

[Steps on HF github](https://github.com/apple/ml-stable-diffusion#-converting-models-to-core-ml)

Step 1: Create a Python environment and install dependencies:

conda create -n coreml_DreamBoothm1 python=3.8 -y
conda activate coreml_DreamBoothm1

-- missing some steps


cd /path/to/cloned/ml-stable-diffusion/repository
pip install -e .

## CompVis
[CompVis](https://github.com/CompVis/stable-diffusion)   

# Converting keras model to coreml
[converting a keras model to coreml](https://heartbeat.comet.ml/using-coremltools-to-convert-a-keras-model-to-core-ml-for-ios-d4a0894d4aba)

## OPTIMIZATIONS FOR MAC

[shaders](https://developer.apple.com/documentation/metal/performing_calculations_on_a_gpu)
Example:
Converting model:
`import torchvision`
`model = torchvision.models.resnet50()`
`model_mps = model.to(device=mpsDevice)`

Run the model
sample_input = torch.randn((32,3,254,254), device=mpsDevice)
prediction = model_mps(sample_input)

### Custom Operations
TF_MetalStream protocol

@protocol TF_MetalStream

- (id <MLTCommandBuffer>)currentCommandBuffer;
- (dispach_queue_t)queue;
- (void)commit;
- (void)commitAndWait;

@end

Three steps to create custom operation
1.  Register

REGISTER_OP("ZeroOut")
    .Input("to_zero: int32")
    .Output("zerod: int32")
    .SetShapeFn([](::tensorflow::show_inference::InferenceContext* c){
        c -> set_output(0, c -> input(0));
        return Status::OK();
    });

### Shared Events

let event = MTLCreateSystemDefualtDevice()!.makeSharedEvent()!
executionDescriptor.signal(event, atExecutionEvent: .completed, value: 1)

New Ops:  Improvements!!
LSTM, RNN, GRU
let descriptor = MPSGraphLSTMDescriptor()

### MaxPooling 
let descriptor = MPSGraphPooling4DOpDescriptor(kernelSizes: @[1,1,3,3],
                                                paddingStyle: .TF_SAME)
descriptor.returnIndicesMode = .globalFlatten4D

let [poolingTensor, indicesTensor] = graph.maxPooling4DReturnIndices(sourceTensor,
                                                                      descriptor: descriptor,
                                                                     name: nil
### Random
let stateTensor = graph.randomPhiloxStateTensor(seed: 2022, name: nil)
let descriptor = MPSGraphRanomOpDescriptor(distribution: .truncatedNormal,
                                           dateType: .float32)
### Tensor Manipulations
let expandedTensor = graph.expandDims(inputTensor,
                                      axis = 1,
                                      name: nil)

let squeezedTensor = graph.squeeze(expandedTensor,
                                    axis = 1,
                                    name: nil)

 let [split1, split2] = graph.split(squeezedTensor,
                                    numSplits: 2,
                                    axis = 0,
                                    name: nil)
                                     

let stackedTensor = graph.stack([split1, split2],
                                 axis = 0,
                                name: nil)








Successfully installed Pillow-9.4.0 accelerate-0.17.0 diffusers-0.14.0 importlib-metadata-6.0.0 numpy-1.23.5 psutil-5.9.4 python-coreml-stable-diffusion-0.1.0 regex-2022.10.31 scipy-1.10.1 tokenizers-0.13.2 torch-1.13.1 transformers-4.26.1 zipp-3.15.0

## Adjusted from sprint noteboobk
`pip install keras_cv===0.4.2 tensorflow_datasets===4.8.1 imutils opencv-python matplotlib pycocotools`

Successfully installed absl-py-1.4.0 click-8.1.3 contourpy-1.0.7 cycler-0.11.0 dill-0.3.6 dm-tree-0.1.8 etils-1.0.0 fonttools-4.39.0 googleapis-common-protos-1.58.0 importlib-resources-5.12.0 imutils-0.5.4 keras_cv-0.4.2 kiwisolver-1.4.4 matplotlib-3.7.1 opencv-python-4.7.0.72 promise-2.3 pycocotools-2.0.6 pyparsing-3.0.9 python-dateutil-2.8.2 six-1.16.0 tensorflow-metadata-1.12.0 tensorflow_datasets-4.8.1 termcolor-2.2.0 toml-0.10.2

# original
pip install tensorflow-macos===2.11.0 tensorflow-metal===0.7.0 keras_cv===0.4.2 tensorflow_datasets===4.8.1 pillow===9.4.0 imutils opencv-python matplotlib pycocotools


## Try this suggestion from @AnimeGuru in Discord
conda create --name DreamboothM1 python=3.8
conda activate DreamboothM1
conda install -c conda-forge tensorflow=2.11.0
pip install tensorflow-macos===2.11.0 tensorflow-metal===0.7.0 keras_cv===0.4.2 tensorflow_datasets===4.8.1 pillow===9.4.0 imutils opencv-python matplotlib pycocotools
conda install -c conda-forge -y pandas jupyter

In the notebook setting use_mp = False & tf.keras.mixed_precision.set_global_policy(None)
Also changing the optimizer

lr = 5e-6
beta_1, beta_2 = 0.9, 0.999
# weight_decay = (1e-2,)
decay = 1e-2
epsilon = 1e-08

optimizer = tf.keras.optimizers.legacy.Adam(
    learning_rate=lr,
    # weight_decay=weight_decay,
    decay=decay,
    beta_1=beta_1,
    beta_2=beta_2,
    epsilon=epsilon,
)

## PROMPTS

[Negative Prompts](https://stable-diffusion-art.com/how-negative-prompt-work/)

Boilder plate for negative prompts
ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face

