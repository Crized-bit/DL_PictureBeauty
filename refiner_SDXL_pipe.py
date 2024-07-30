import torch
import cv2
import numpy as np
import os
from tqdm import tqdm
from PIL.Image import fromarray
from diffusers import StableDiffusionXLControlNetPipeline, StableDiffusionXLImg2ImgPipeline # type: ignore
from diffusers import ControlNetModel, DPMSolverMultistepScheduler   # type: ignore
from utils.preprocessors import tile_resample_processor
from utils.embedders import get_compel_embeddings, get_longclip_embeds

# Default parameters block
DEVICE = torch.device("cuda:2")
output_folder = "/home/Diffusion_VK/results/refined_sasha_model/"
control_folder = "/home/Diffusion_VK/results/baseline/"

# Creating a pipe
pipeline = StableDiffusionXLImg2ImgPipeline.from_single_file(
    # "/home/web_ui/stable-diffusion-webui/models/Stable-diffusion/cyberrealisticXL_v22.safetensors",
    "/home/web_ui/stable-diffusion-webui/models/Stable-diffusion/jibMixRealisticXL_v160Aphrodite.safetensors",
    use_safetensors=True,
    torch_dtype=torch.bfloat16)

pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config,
                                                 algorithm_type="sde-dpmsolver++",
                                                 final_sigmas_type = "zero", 
                                                 timestep_spacing="linspace", 
                                                #  use_exponential_sigmas=True,
                                                 do_brownian_noise=True,
                                                 use_karras_sigmas=True,
                                                            )


# Offload to GPU
pipeline.to(DEVICE)

# Create embedder instance to generate embeddings
embedder = get_compel_embeddings(pipeline=pipeline,
                                 device=DEVICE)
# Dataset inference
for filename in tqdm(sorted(os.listdir(control_folder))):
    # if filename[:-4] not in ["gothica"]:
    #     continue
    img = cv2.imread(os.path.join(control_folder, filename))
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    controlnet_input = fromarray(img)

    # Generate image
    images = []
    if os.path.isfile(os.path.join("/home/dataset/Games_captions", filename[:-4] + '.txt')):
        with open("/home/dataset/Games_captions/" + filename[:-4] + '.txt', "r", encoding="utf-8") as f:
            positive_promt = f.read()
    else:
        continue

    negative_promt = 'bad eyes, cgi, airbrushed, plastic, deformed, watermark, polygons, inaccurate body, worst quality, low quality, normal quality, jpeg artifacts, unrealistic, flat, triangular, low resolution, bad composition'

    con_embeds, pooled_prompt_embeds, neg_embeds, negative_pooled_prompt_embeds = embedder(positive_promt, 
                                                                                           negative_promt)

    # Generate image
    images = []
    # images.append(np.array(controlnet_input))
    for strength in [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
        result_img = pipeline(prompt_embeds=con_embeds,
                            pooled_prompt_embeds = pooled_prompt_embeds,
                            negative_prompt_embeds=neg_embeds,
                            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds,
                            image=controlnet_input,
                            generator=torch.Generator(device=DEVICE).manual_seed(1787813600),
                            num_inference_steps = 30,
                            strength = strength,
                            guidance_scale = 3.5,
                            ).images[0] # type: ignore
    
    
        images.append(np.array(result_img))
    # Then prepare list of np.arrays' to hcat, then do it
    cv2.imwrite(output_folder + filename, cv2.cvtColor(cv2.hconcat(images), cv2.COLOR_RGB2BGR))