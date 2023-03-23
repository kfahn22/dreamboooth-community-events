## Loss scale optimizer error
[loss scale optimizer](https://keras.io/api/mixed_precision/loss_scale_optimizer/#baselossscaleoptimizer-class)

[github](https://github.com/keras-team/keras/blob/v2.11.0/keras/mixed_precision/loss_scale_optimizer.py#L361-L586)

[adam](https://github.com/keras-team/keras/blob/v2.11.0/keras/optimizers/legacy/adam.py)

## Keras Examples
[keras examples](https://keras.io/examples/)
[keras_cv](https://github.com/kfahn22/keras-cv/tree/master/keras_cv/models/stable_diffusion)
[Fine-tuning stable-diffusion](https://keras.io/examples/generative/finetune_stable_diffusion/)

## NOTES on Keras model

### Legacy Adam optimizer implementation.

from keras.optimizers.optimizer_v2 import adam

class Adam(adam.Adam):
    pass

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
-experimental version of numpy
[numpy](https://www.tensorflow.org/api_docs/python/tf/experimental/numpy)
-used in lines 134 - 136
`attention_mask = tfnp.triu(
                tf.ones((1, 1, length, length), dtype=self.compute_dtype)
                * -tfnp.inf,
                k=1,
            )`
- triu: Return a copy of a matrix with the elements below the k-th diagonal zeroed.

[pytorch triu](https://pytorch.org/docs/stable/generated/torch.triu.html)
`torch.triu(input, diagonal=0, *, out=None)`

## Errors

## backend.py
[error re dtype](https://github.com/keras-team/keras/blob/master/keras/backend.py)
