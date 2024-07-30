import torch
import cv2
import numpy as np
import os
from tqdm import tqdm
from PIL.Image import fromarray
from diffusers import StableDiffusionXLControlNetPipeline # type: ignore
from diffusers import ControlNetModel, DPMSolverMultistepScheduler   # type: ignore
from utils.preprocessors import tile_resample_processor
from utils.embedders import get_compel_embeddings, get_longclip_embeds

# Default parameters block
DEVICE = torch.device("cuda:2")
output_folder = "/home/dataset/Train_data/output_pictures/"
initial_folder = "/home/dataset/Train_data/initial_pictures/"
control_folder = "/home/dataset/Train_data/processed_pictures/"
captions_folder= "/home/dataset/Train_data/captions/"

# ControlNets block
controlnets = ControlNetModel.from_pretrained(
        "/home/web_ui/stable-diffusion-webui/models/ControlNet/controlnet_tile", 
        torch_dtype=torch.bfloat16, use_safetensors=True)

# Creating a pipe
pipeline = StableDiffusionXLControlNetPipeline.from_single_file(
    # "/home/web_ui/stable-diffusion-webui/models/Stable-diffusion/cyberrealisticXL_v22.safetensors",
    "/home/web_ui/stable-diffusion-webui/models/Stable-diffusion/juggernautXL_juggXIByRundiffusion.safetensors",
    # "/home/web_ui/stable-diffusion-webui/models/Stable-diffusion/jibMixRealisticXL_v160Aphrodite.safetensors",
    use_safetensors=True,
    torch_dtype=torch.bfloat16, 
    controlnet = controlnets)

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
for filename in tqdm(sorted(os.listdir(initial_folder))):
    img = cv2.imread(os.path.join(initial_folder, filename))
    
    img = tile_resample_processor(img=img, 
                                  thr_a=4.)
    controlnet_input = fromarray(img)
    if control_folder is not None:
        controlnet_input.save(control_folder + filename)
    # Generate image
    images = []
    if os.path.isfile(os.path.join(captions_folder, filename[:-4] + '.txt')):
        with open(captions_folder + filename[:-4] + '.txt', "r", encoding="utf-8") as f:
            positive_promt = f.read()
    else:
        print(f"Didnt find caption for {filename}")
        continue

    negative_promt = 'bad eyes, cgi, airbrushed, plastic, deformed, watermark, polygons, inaccurate body, worst quality, low quality, normal quality, jpeg artifacts, unrealistic, flat, triangular, low resolution, bad composition'

    con_embeds, pooled_prompt_embeds, neg_embeds, negative_pooled_prompt_embeds = embedder(positive_promt, 
                                                                                           negative_promt)

    # Generate image
    images = []
    # images.append(np.array(controlnet_input))
    result_img = pipeline(prompt_embeds=con_embeds,
                          pooled_prompt_embeds = pooled_prompt_embeds,
                          negative_prompt_embeds=neg_embeds,
                          negative_pooled_prompt_embeds = negative_pooled_prompt_embeds,
                          image=controlnet_input,
                          generator=torch.Generator(device=DEVICE).manual_seed(1787813600),
                          num_inference_steps = 30,
                          controlnet_conditioning_scale = 0.5,
                          guidance_scale = 5.,
                          ).images[0] # type: ignore
    
    
    images.append(np.array(result_img))
    # Then prepare list of np.arrays' to hcat, then do it
    cv2.imwrite(output_folder + filename, cv2.cvtColor(cv2.hconcat(images), cv2.COLOR_RGB2BGR))