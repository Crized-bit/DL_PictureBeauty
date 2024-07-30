import torch
import cv2
import numpy as np
import os
from tqdm import tqdm
from PIL.Image import fromarray
from diffusers import FluxImg2ImgPipeline # type: ignore
from diffusers import FluxTransformer2DModel, FluxPipeline
from transformers import T5EncoderModel
from sd_embed.embedding_funcs import get_weighted_text_embeddings_flux1
from optimum.quanto import freeze, qfloat8, quantize
from torchao.quantization import quantize_, int8_weight_only

# Default parameters block
DEVICE = torch.device("cuda:2")
output_folder = "/home/dataset/Train_data/refined_pictures/"
control_folder = "/home/dataset/Train_data/output_pictures/"
captions_folder= "/home/dataset/Train_data/captions/"

bfl_repo = "black-forest-labs/FLUX.1-dev"
dtype = torch.bfloat16

# model_path = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/black-forest-labs/FLUX.1-dev_main"

transformer = FluxTransformer2DModel.from_pretrained(
    bfl_repo
    , subfolder = "transformer"
    , torch_dtype = torch.bfloat16
)

quantize_(transformer, int8_weight_only())

pipeline = FluxImg2ImgPipeline.from_pretrained(bfl_repo, transformer=transformer, torch_dtype=dtype)

# Offload to GPU
pipeline.to(DEVICE)

# Dataset inference
for filename in tqdm(sorted(os.listdir(control_folder))):
    # if filename[:-4] not in ["gothica"]:
    #     continue
    img = cv2.imread(os.path.join(control_folder, filename))
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _  = img.shape
    controlnet_input = fromarray(img)

    # Generate image
    images = []
    if os.path.isfile(os.path.join(captions_folder, filename[:-4] + '.txt')):
        with open(captions_folder + filename[:-4] + '.txt', "r", encoding="utf-8") as f:
            positive_promt = f.read()
    else:
        continue

    negative_promt = 'bad eyes, cgi, airbrushed, plastic, deformed, watermark, polygons, inaccurate body, worst quality, low quality, normal quality, jpeg artifacts, unrealistic, flat, triangular, low resolution, bad composition'

    prompt_embeds, pooled_prompt_embeds = get_weighted_text_embeddings_flux1(
        pipe = pipeline,
        prompt = positive_promt,
        device = DEVICE,
    )
    
    # Generate image
    images = []
    # images.append(np.array(controlnet_input))
    result_img = pipeline(prompt_embeds=prompt_embeds,
                            pooled_prompt_embeds = pooled_prompt_embeds,
                            image=controlnet_input,
                            generator=torch.Generator(device=DEVICE).manual_seed(1787813600),
                            num_inference_steps = 50,
                            strength = 0.1,
                            guidance_scale = 4.0,
                            height = height,
                            width = width,
                            ).images[0] # type: ignore
    
    
    images.append(np.array(result_img))
    # Then prepare list of np.arrays' to hcat, then do it
    cv2.imwrite(output_folder + filename, cv2.cvtColor(cv2.hconcat(images), cv2.COLOR_RGB2BGR))