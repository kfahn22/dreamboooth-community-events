# DREAMBOOTH

## LOADING THE DATA

[tutorial](https://pytorch.org/data/main/dp_tutorial.html)
[interable datapipes](https://pytorch.org/data/main/torchdata.datapipes.iter.html)
[binary file loader](https://pytorch.org/data/main/generated/torchdata.datapipes.iter.Bz2FileLoader.html#torchdata.datapipes.iter.Bz2FileLoader)

## 
[readme for diffusers](https://github.com/huggingface/diffusers)
`pip install --upgrade diffusers[torch]`

## Prepare the images

### augmenter

[](https://pytorch.org/vision/stable/transforms.html)
[center crop](https://pytorch.org/vision/main/generated/torchvision.transforms.CenterCrop.html)
`torchvision.transforms.CenterCrop(size)`
[random flip](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomHorizontalFlip.html)
`torchvision.transforms.RandomHorizontalFlip(p=0.5)`

-- rescaling
[rescaling](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Rescaling)
[tutorial](https://www.tutorialspoint.com/pytorch-how-to-resize-an-image-to-a-given-size)
[resize](https://pytorch.org/vision/stable/generated/torchvision.transforms.Resize.html#torchvision.transforms.Resize)
`torchvision.transforms.Resize(size, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias='warn')`

-layers
[PaddedConv2D](https://github.com/keras-team/keras-cv/blob/master/keras_cv/models/stable_diffusion/__internal__/layers/padded_conv2d.py)

-schedulers
(default pytorch)[https://huggingface.co/docs/diffusers/v0.3.0/en/api/schedulers]

Source for noise_scheduler.py
[ddmp](https://github.com/huggingface/diffusers/blob/v0.3.0/src/diffusers/schedulers/scheduling_ddpm.py)
[adding noise](https://github.com/huggingface/diffusers/blob/v0.3.0/src/diffusers/schedulers/scheduling_karras_ve.py#L115)

GradientTape()
[autograd](https://pytorch.org/tutorials/beginner/former_torchies/autograd_tutorial.html)

Create a tensor and set requires_grad=True to track computation with it

## Changes in Pytorch 2.0

PyTorch 2.0 includes a scaled dot-product attention function as part of torch.nn.functional

-- doesn't work with mps
`torch.compile()`