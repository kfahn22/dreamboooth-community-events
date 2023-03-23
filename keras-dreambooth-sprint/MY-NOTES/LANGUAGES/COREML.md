# Diffusion on COREML

[coreml](https://developer.apple.com/documentation/coreml)
[apple-github](https://github.com/apple/ml-stable-diffusion)
[stable-diffusion-coreml](https://machinelearning.apple.com/research/stable-diffusion-coreml-apple-silicon) 

[Steps on HF github](https://github.com/apple/ml-stable-diffusion#-converting-models-to-core-ml


## Articles

[article](https://www.chrisjmendez.com/2023/01/07/run-stable-diffusion-on-macbook-pro-m1-core-ml/)
[diffuers](https://pypi.org/project/diffusers/)

## Hugging Face

[Hugging Face coreml blog](https://huggingface.co/blog/diffusers-coreml)  
[swift](https://github.com/huggingface/swift-coreml-diffusers)



`pip install huggingface_hub`
`pip install git+https://github.com/apple/ml-stable-diffusion`


from huggingface_hub import snapshot_download
from huggingface_hub.file_download import repo_folder_name
from pathlib import Path
import shutil

repo_id = "apple/coreml-stable-diffusion-v1-4"
variant = "original/packages"

def download_model(repo_id, variant, output_dir):
    destination = Path(output_dir) / (repo_id.split("/")[-1] + "_" + variant.replace("/", "_"))
    if destination.exists():
        raise Exception(f"Model already exists at {destination}")
    
    # Download and copy without symlinks
    downloaded = snapshot_download(repo_id, allow_patterns=f"{variant}/*", cache_dir=output_dir)
    downloaded_bundle = Path(downloaded) / variant
    shutil.copytree(downloaded_bundle, destination)

    # Remove all downloaded files
    cache_folder = Path(output_dir) / repo_folder_name(repo_id=repo_id, repo_type="model")
    shutil.rmtree(cache_folder)
    return destination

model_path = download_model(repo_id, variant, output_dir="./models")
print(f"Model downloaded at {model_path}")

python -m python_coreml_stable_diffusion.pipeline \
          -i ~/Documents/AI_MODELS \
          -o ~/Desktop/my-images \
          --compute-unit CPU_AND_NE \
          --seed <enter a 3-4 digit number> \
          --prompt <enter your prompt here>
