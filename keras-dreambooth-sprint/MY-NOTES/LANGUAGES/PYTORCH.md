# PYTORCH

## Docs
[pytorch-github](https://github.com/pytorch/pytorch/wiki/)

## Pytorch using mps
[pytorch-backend](https://pytorch.org/docs/master/notes/mps.html)

### mps backend

if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

else:
    mps_device = torch.device("mps")

    # Create a Tensor directly on the mps device
    x = torch.ones(5, device=mps_device)
    # Or
    x = torch.ones(5, device="mps")

    # Any operation happens on the GPU
    y = x * 2

    # Move your model to mps just like any other device
    model = YourFavoriteNet()
    model.to(mps_device)

    # Now every call runs on the GPU
    pred = model(x)

## Stable Diffusion 
[pytorch implementation](https://github.com/kfahn22/stable-diffusion-pytorch/tree/main/stable_diffusion_pytorch)

[blog](https://pytorch.org/blog/accelerated-diffusers-pt-20/)



## LINKS
[autograd](https://pytorch.org/docs/master/autograd.html)
[haven't read](https://towardsdatascience.com/recreating-keras-code-in-pytorch-an-introductory-tutorial-8db11084c60c)
[layers](https://pytorch.org/docs/stable/search.html?q=layers&check_keywords=yes&area=default)


from diffusers import StableDiffusionPipeline
from diffusers.models.cross_attention import AttnProcessor2_0

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe.to("cuda")
pipe.unet.set_attn_processor(AttnProcessor2_0())

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]



[pipeline](https://pytorch.org/docs/master/pipeline.html)

# Build pipe.
fc1 = nn.Linear(16, 8).cuda(0)
fc2 = nn.Linear(8, 4).cuda(1)
model = nn.Sequential(fc1, fc2)
model = Pipe(model, chunks=8)
input = torch.rand(16, 16).cuda(0)
output_rref = model(input)

## CONVERSIONS

### Moving from keras to pytorch
[Moving from keras to pytorch](https://towardsdatascience.com/moving-from-keras-to-pytorch-f0d4fff4ce79)
[Kaggle Notebook](https://www.kaggle.com/code/mlwhiz/third-place-model-for-toxic-comments-in-pytorch/notebook)

### Moving from tensorflow to pytorch
[Moving](https://neptune.ai/blog/moving-from-tensorflow-to-pytorch)
[arrays](https://www.tutorialspoint.com/how-to-convert-a-numpy-ndarray-to-a-pytorch-tensor-and-vice-versa)
[images to tensors](https://towardsdatascience.com/convert-images-to-

### Converting data to torch
[dataset conversion](https://stackoverflow.com/questions/67345480/converting-a-tf-dataset-to-a-pytorch-dataset)

`import tensorflow datasets as tfds import torch.nn as nn`  
`def train dataloader (batch size):`  
`return tfds.as_numpy(tfds.load('mnist').batch(batch_size))`  
`class Model (n.Module):`  
`def forward(self, x):`  
`× = torch. as_tensor (x, device='mps')`  

## Tensorflow and pytorch
[tf and pytorch](https://stackoverflow.com/questions/74704866/running-tf-and-torch-on-one-virtual-environment)
[pytorh vs nightly](https://discuss.pytorch.org/t/pytorch-nightly-vs-stable/105633)
[torch on mps](https://pytorch.org/docs/master/notes/mps.html)

## ISSUES

### Precision
[github pytorch](https://github.com/pytorch/pytorch)
Mixed precision??
[github isssue](https://github.com/pytorch/pytorch/issues/88415)
[Bias](https://stackoverflow.com/questions/55229636/import-lstm-from-tensorflow-to-pytorch-by-hand?rq=1)
[AMX?](https://discuss.pytorch.org/t/how-to-check-mps-availability/152015/9)
[seed](https://stackoverflow.com/questions/74614882/manual-seed-for-mps-devices-in-pytorch)
`import torch` 
`torch.manual_seed(0)`

### Optimization and Training

ops
[torch ops](https://github.com/pytorch/pytorch/blob/master/torch/_ops.py)
[pytorch-lightning](https://pytorch-lightning.readthedocs.io/en/stable/accelerators/mps_basic.html)
`PYTORCH_ENABLE_MPS_FALLBACK=1 python your_script.py `

to enable training on GPUS
`trainer = Trainer(accelerator="mps", devices=1)`



## FUNDAMENTALS

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

## We can specify the device at the moment of creation - RECOMMENDED!
torch.manual_seed(42)
a = torch.randn(1, requires_grad=True, type=torch. float, device=device)

## XLA
[XLA](https://github.com/pytorch/xla)
`pip3 install torch_xla[tpuvm]`

- [pipeline.py](https://github.com/kjsman/stable-diffusion-pytorch/blob/main/stable_diffusion_pytorch/pipeline.py)

## Class on Torch
[Lecture](http://kiwi.bridgeport.edu/cpeg589/CPEG589_Lecture4.pdf)