# Notes on Using Hugging Face

## Installation
[installation](https://huggingface.co/docs/huggingface_hub/installation)

## Logging In
[hugging face-cli](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/login)

-- I get an error when I try to run this command

`pip install 'huggingface_hub[cli,torch]'`
[login issues](https://discuss.huggingface.co/t/huggingface-cli-login/25567/2)
[login in error](https://discuss.huggingface.co/t/how-to-login-to-huggingface-hub-with-access-token/22498/13)
python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('MY_HUGGINGFACE_TOKEN_HERE')"

## Working on a m1
[hg-working on m1 mac](https://huggingface.co/docs/diffusers/optimization/mps)

## Hugging Face and Pytorch

[Install torch with stable diffusion](https://huggingface.co/spaces/stabilityai/stable-diffusion/discussions/1835)
[corgi](https://huggingface.co/dreambooth-hackathon/ccorgi-dog)
[diffusors-pytorch](https://huggingface.co/search/full-text?q=diffusers%5Btorch%5D)

## Hugging Face coreml
[Hugging-Face-conversion](https://huggingface.co/blog/diffusers-coreml)
`pip install huggingface_hub`
`pip install git+https://github.com/apple/ml-stable-diffusion`
[coverted](https://huggingface.co/apple/coreml-stable-diffusion-2-base)
[stable-diffusion-coreml](https://machinelearning.apple.com/research/stable-diffusion-coreml-apple-silicon) 

[Steps on HF github](https://github.com/apple/ml-stable-diffusion#-converting-models-to-core-ml)

[metal](https://developer.apple.com/metal/pytorch/)
[](https://huggingface.co/docs/diffusers/training/dreambooth)

### Passing in a local folder
`from diffusers import DiffusionPipeline`

`repo_id = "./stable-diffusion-v1-5"`
`stable_diffusion = DiffusionPipeline.from_pretrained(repo_id)`

## Steps
### First initialize a new environment 
`conda create --name Dreambooth-env python=3.8`
`conda activate Dreambooth-env`

### Clone diffusers repo
`git clone https://github.com/huggingface/diffusers`
`cd diffusers`
`pip install -e .`
`pip install -U -r ./examples/dreambooth/requirements.txt`


### Add requirements outside of notebook
`conda install pip`
`conda install ipykernel`
`conda install pytorch torchvision torchaudio -c pytorch-nightly`
`conda install -c conda-forge huggingface_hub`


Ran in terminal Asked a bunch of questions
`accelerate config`
accelerate configuration saved at /Users/kathrynfahnline/.cache/huggingface/accelerate/default_config.yaml 

### In notebook
`pip install -r requirements.txt`

### tutorial
[dataquest](https://www.dataquest.io/live-tutorials-and-project-walkthroughs/)

### training Stable Diffusion
[training](https://huggingface.co/docs/diffusers/main/en/tutorials/basic_training)

[dreambooth-tutorial-windows](https://www.youtube.com/watch?v=w6PTviOCYQY)

[model](https://modal.com/docs/guide/ex/dreambooth_app)

[accelerate](https://github.com/huggingface/accelerate/)
## Discussion

[Diffusers discussion](https://discuss.huggingface.co/c/discussion-related-to-httpsgithubcomhuggingfacediffusers/63)
