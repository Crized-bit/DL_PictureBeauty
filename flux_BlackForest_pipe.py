import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["FLUX_DEV_DEPTH_LORA"] = "/home/Diffusion_VK/weights/flux1-depth-dev-lora.safetensors"

from torchao.quantization import quantize_, int8_weight_only
from flux.util import load_clip, load_ae, load_flow_model, load_t5, load_sft
from flux.sampling import prepare, denoise, get_noise, get_schedule
from pathlib import Path
from torch import Tensor
import numpy as np
import PIL.Image as Image
import torch
from einops import rearrange, repeat
import math
from tqdm import tqdm
import cv2
from utils.preprocessors import tile_resample_processor

output_folder = "/home/Diffusion_VK/results/final_flux_after_6500_uniform/"
control_folder = "/home/dataset/Games/"
captions_folder= "/home/dataset/Games_captions/"

def prepare_control(
    t5,
    clip,
    img: Tensor,
    prompt: str | list[str],
    ae,
    conditional_image,
) -> dict[str, Tensor]:
    # load and encode the conditioning image
    bs, _, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img_cond = np.array(conditional_image)
    img_cond = torch.from_numpy(img_cond).float() / 127.5 - 1.0
    img_cond = rearrange(img_cond, "h w c -> 1 c h w")

    with torch.no_grad():
        img_cond = ae.encode(img_cond.to(device=img.device))

    img_cond = img_cond.to(torch.bfloat16)
    img_cond = rearrange(img_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img_cond.shape[0] == 1 and bs > 1:
        img_cond = repeat(img_cond, "1 ... -> bs ...", bs=bs)

    return_dict = prepare(t5, clip, img, prompt)
    return_dict["img_cond"] = img_cond
    return return_dict

def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )


t5 = load_t5("cuda", max_length=512)
clip = load_clip("cuda")
ae = load_ae("flux-dev-depth-lora", device="cuda")

model = load_flow_model("flux-dev-depth-lora", device="cpu")

for _, module in model.named_modules():
     if hasattr(module, "set_scale"):
         module.set_scale(0.85)

def filter_fn(module: torch.nn.Module, fqn: str) -> bool:
        if not hasattr(module, "weight"):
            return False
        
        if module.weight is None:
            return False
        return True
    
quantize_(model, int8_weight_only(), filter_fn=filter_fn)
model.to('cuda')

for filename in tqdm(sorted(os.listdir(control_folder))):
    # if filename[:-4] not in ["gothica"]:
    #     continue
    img = cv2.imread(os.path.join(control_folder, filename))
    
    img = tile_resample_processor(img=img, 
                                  thr_a=4.)
    controlnet_input = Image.fromarray(img)

    # Generate image
    images = []
    if os.path.isfile(os.path.join(captions_folder, filename[:-4] + '.txt')):
        with open(captions_folder + filename[:-4] + '.txt', "r", encoding="utf-8") as f:
            positive_promt = f.read()
    else:
        continue

    negative_promt = 'bad eyes, cgi, airbrushed, plastic, deformed, watermark, polygons, inaccurate body, worst quality, low quality, normal quality, jpeg artifacts, unrealistic, flat, triangular, low resolution, bad composition'

    w, h = controlnet_input.size

    with torch.no_grad():
        x = get_noise(
            1,
            h,
            w,
            device=torch.device('cuda'),
            dtype=torch.bfloat16,
            seed=1337,
        )

        inp = prepare_control(
            t5,
            clip,
            x,
            prompt=positive_promt,
            ae=ae,
            conditional_image=controlnet_input)
        
        timesteps = get_schedule(50, inp["img"].shape[1], shift=True)


        x = denoise(model, **inp, timesteps=timesteps, guidance=10.)

        x = unpack(x.float(), h, w)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            x = ae.decode(x)

    x = x.clamp(-1, 1)
    x = rearrange(x[0], "c h w -> h w c")
    img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

    images.append(np.array(img))
    # Then prepare list of np.arrays' to hcat, then do it
    cv2.imwrite(output_folder + filename, cv2.cvtColor(cv2.hconcat(images), cv2.COLOR_RGB2BGR))