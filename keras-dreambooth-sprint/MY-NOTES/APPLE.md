# APPLE DOCUMENTATION AND RESOURCES

[Apple research](https://machinelearning.apple.com)
[Resources](https://developer.apple.com/machine-learning/resources/)
[WWWDC22](https://developer.apple.com/videos/all-videos/)
[metal-code](https://developer.apple.com/metal/sample-code/)

## tensorflow-metal

[tensorflow-metal](https://developer.apple.com/metal/tensorflow-plugin/)
`bash ~/miniconda.sh -b -p $HOME/miniconda`  
`source ~/miniconda/bin/activate`
`conda install -c apple tensorflow-deps`

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

## Lightning

[lightning](https://lightning.ai/docs/pytorch/stable/starter/installation.html)
--needed for M1/M2/M3
`export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1`
`export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1`

`python -m pip install -U lightning`

`conda activate my_env`
`conda install pytorch-lightning -c conda-forge`

`trainer = Trainer(accelerator="mps", devices=1)`

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

## DIFFUSERS

[Apple-ml-stable-diffusion](https://github.com/apple/ml-stable-diffusion)
[stability-ai](https://github.com/Stability-AI/stablediffusion)
[CompVis](https://github.com/CompVis/stable-diffusion)
[hugging-face]( https://huggingface.co/docs/diffusers/quicktour)
[Another repo](https://github.com/kjsman/stable-diffusion-pytorch)
[](https://github.com/TheLastBen/fast-stable-diffusion)


### Stable Diffusion

[Apple m1 on mac](https://wandb.ai/morgan/stable-diffusion/reports/Running-Stable-Diffusion-on-an-Apple-M1-Mac-With-HuggingFace-Diffusers--VmlldzoyNTU2ODc2)

## Hugging Face

[hg-working on m1 mac](https://huggingface.co/docs/diffusers/optimization/mps)

### Hugging Face coreml

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

## Converting keras model to coreml

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

## Custom Operations

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

## Issues

this is from 12/2021 maybe fixed??
[arm64](https://developer.apple.com/forums/thread/695963)

Requirements: 
- arm64 version of Python
- Pytorch 1.13

[speed issues](https://github.com/pytorch/pytorch/issues/77799)

## Misc

[fastai class](https://github.com/fastai/course22p2/tree/master/nbs)

[h5py](https://docs.h5py.org/en/stable/build.html)
