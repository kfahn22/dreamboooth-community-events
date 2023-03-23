# DREAMBOOTH

## Github repos

### CompVis
[CompVis](https://github.com/CompVis/stable-diffusion)   

## GitHub repos working on training DreamBooth on Mac
[XavierXiao](https://github.com/XavierXiao/Dreambooth-Stable-Diffusion/pull/36)
[DreanBoothMac](https://github.com/SujeethJinesh/DreamBoothMac)
[Issues](https://github.com/CompVis/stable-diffusion/issues/25)

## Link to notebook explaining how to use Lambda in Colab
[notebook](https://github.com/carolineechen/hf-community-events/blob/main/keras-dreambooth-sprint/runhouse/dreambooth_rh_colab.ipynb)

## LOADING THE DATA

[tutorial](https://pytorch.org/data/main/dp_tutorial.html)
[interable datapipes](https://pytorch.org/data/main/torchdata.datapipes.iter.html)
[binary file loader](https://pytorch.org/data/main/generated/torchdata.datapipes.iter.Bz2FileLoader.html#torchdata.datapipes.iter.Bz2FileLoader)

## 
[readme for diffusers](https://github.com/huggingface/diffusers)
`pip install --upgrade diffusers[torch]`

## Prepare the images

# Using SimpleTokenizer from Keras to load model from this site.
[clip](https://github.com/openai/CLIP)

# Prepare the inputs
image, class_id = cifar100[3637]
image_input = preprocess(image).unsqueeze(0).to(device)
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)


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

## PROMPTS

[Negative Prompts](https://stable-diffusion-art.com/how-negative-prompt-work/)

Boilder plate for negative prompts
ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face
