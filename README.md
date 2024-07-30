# SDXL + FLUX image beautifier 

This project is aimed to train or inference Diffusion models to achive low-quality photo restoration without composition and detail loss.

## How to install

1) Clone this repo and install PyTorch (I used python 3.10)
2) If you want to use my features with brownian noise, you should install [my fork of diffusers libriary](https://github.com/Crized-bit/diffusers) as editable installation (because they didn't want to merge for some reasons) 
3) Install Flux repo that i use [FLUX .1 dev](https://github.com/Crized-bit/flux)
4) Also for my experiments i'm using [Compel](https://github.com/damian0815/compel), consider installing it too! 
5) Optional. If you are interested in testing [LongClip](https://github.com/beichenzbc/Long-CLIP), also install it or comment out
6) Install https://github.com/xhinker/sd_embed, works the same as Compel, but usable with FLUX
7) Download [juggernautXL_juggXIByRundiffusion](https://civitai.com/models/133005/juggernaut-xl?modelVersionId=782002) model from CivAI somewhere, because my scripts use that model

## How to use

It's research repo, but some things work out of box. For example, if you want to generate better version of your "LQ" photo, consider using that scripts:
1) [baseline_brownian_karras.py](https://github.com/Crized-bit/DL_PictureBeauty/blob/main/baseline_brownian_karras.py) - that code is without CLI, but it's easy to understand. It will generate "HQ" image with SDXL but without refiner
2) You may do refinement with [flux_diffusers_pipe.py](https://github.com/Crized-bit/DL_PictureBeauty/blob/main/flux_diffusers_pipe.py)

All scripts use "Path headers" such as:
```python
output_folder = "/home/dataset/Train_data/refined_pictures/"
control_folder = "/home/dataset/Train_data/output_pictures/"
captions_folder= "/home/dataset/Train_data/captions/"
```
You should consider changing them, as well as DEVICE env and PATH in those codelines: 
1) https://github.com/Crized-bit/DL_PictureBeauty/blob/6b9bd4ec93f6c04765589764845cade2f06b8b25/baseline_brownian_karras.py#L25-L31
2) https://github.com/Crized-bit/DL_PictureBeauty/blob/6b9bd4ec93f6c04765589764845cade2f06b8b25/baseline_brownian_karras.py#L20-L22

## Training scripts

I've tried to distill that pipeline, but it didn't work :( 

Scripts in that [folder](https://github.com/Crized-bit/DL_PictureBeauty/tree/e3dfa68896f3e87b7cf3357fbb8a4f27b257c61b/training_scripts) are "fixed" version of [diffusers examples](https://github.com/huggingface/diffusers/tree/328e0d20a7b996f9bdb04180457eb08c1b42a76e/examples/flux-control). You can look up [SDXL training](https://github.com/huggingface/diffusers/tree/main/examples/controlnet) as well. Exaples of .sh scipts are provided.

Main complications in traing were:
1) SDXL Controlnet on 1000+ pictures gives bad results
2) FLUX Lora Controlnet (as they released in [their HF repo](https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev-lora)) training was giving once again bad results
3) FLUX original Controlnet (as done in [X-Labs repo](https://github.com/XLabs-AI/x-flux/blob/main/train_flux_deepspeed_controlnet.py)) training was not working at all with LogNorm (0,1) dist, also tried U[0,1]

## Reusable changes:

1) Check latest commit in [FLUX .1 dev](https://github.com/Crized-bit/flux), there's gradient checkpoiting implemented for FLUX training
