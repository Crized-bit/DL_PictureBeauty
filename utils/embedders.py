from compel import Compel, ReturnedEmbeddingsType
import torch
import torch.nn as nn
from collections.abc import Callable
from utils.long_clip.open_clip_long import factory as open_clip
from typing import Callable
from utils.long_clip.model import longclip

def get_compel_embeddings(pipeline, device: torch.device) -> Callable[[str, str], tuple]:
    compel=Compel(tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2],
                            text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],
                            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, 
                            requires_pooled=[False, True],
                            truncate_long_prompts=False,
                            device=device) # type: ignore
    
    def return_embeddings(positive_prompt: str, negative_prompt: str):
        conditioning, pooled_prompt_embeds=compel(positive_prompt)
        negative_conditioning, negative_pooled_prompt_embeds=compel(negative_prompt)
        [con_embeds, neg_embeds] = compel.pad_conditioning_tensors_to_same_length([conditioning, negative_conditioning])

        return con_embeds, pooled_prompt_embeds, neg_embeds, negative_pooled_prompt_embeds

    return return_embeddings
    

def get_longclip_embeds(pipeline, device: torch.device) -> Callable[[str, str], tuple]:
    bigG_model, _, _ = open_clip.create_model_and_transforms(
        'ViT-bigG-14', 
        pretrained='/home/Diffusion_VK/utils/long_clip/checkpoints/open_clip_pytorch_model.bin'
    )
    bigG_model = kps(bigG_model)
    bigG_model.eval().to(device)
    bigG_encoder = bigG_model.encode_text_full

    vitl_model, _ = longclip.load("/home/Diffusion_VK/utils/long_clip/checkpoints/longclip-L.pt", 
                                  device=device)
    vitl_model.eval()
    vitL_encoder = vitl_model.encode_text_full

    openclip_tokenizer = open_clip.get_tokenizer('ViT-bigG-14')

    tokenizers = [longclip.tokenize, openclip_tokenizer]

    text_encoders = [vitL_encoder, bigG_encoder]
    
    def encode_prompts(positive_prompt:str, negative_prompt: str):
        prompts, negative_prompts = 2*[positive_prompt], 2*[negative_prompt]
        
        nonlocal tokenizers
        nonlocal text_encoders

        prompt_embeds_list = []
        for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):

                text_input_ids = tokenizer(prompt)
                prompt_embeds = text_encoder(text_input_ids.to(device))

                prompt_embeds_list.append(prompt_embeds)
        
        pooled_prompt_embeds = bigG_model.encode_text(text_input_ids.to(device))
        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        negative_prompt_embeds_list = []
        for prompt, tokenizer, text_encoder in zip(negative_prompts, tokenizers, text_encoders):

                text_input_ids = tokenizer(prompt)
                negative_prompt_embeds = text_encoder(text_input_ids.to(device))

                negative_prompt_embeds_list.append(negative_prompt_embeds)
        
        negative_pooled_prompt_embeds = bigG_model.encode_text(text_input_ids.to(device))
        negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)
    
        return prompt_embeds, pooled_prompt_embeds, negative_prompt_embeds, negative_pooled_prompt_embeds
    return encode_prompts

def kps(model):
    positional_embedding_pre = model.positional_embedding       
    length, dim = positional_embedding_pre.shape
    keep_len = 20
    posisitonal_embedding_new = torch.zeros([4*length-3*keep_len, dim])
    for i in range(keep_len):
        posisitonal_embedding_new[i] = positional_embedding_pre[i]
    for i in range(length-1-keep_len):
        posisitonal_embedding_new[4*i + keep_len] = positional_embedding_pre[i + keep_len]
        posisitonal_embedding_new[4*i + 1 + keep_len] = 3*positional_embedding_pre[i + keep_len]/4 + 1*positional_embedding_pre[i+1+keep_len]/4
        posisitonal_embedding_new[4*i + 2+keep_len] = 2*positional_embedding_pre[i+keep_len]/4 + 2*positional_embedding_pre[i+1+keep_len]/4
        posisitonal_embedding_new[4*i + 3+keep_len] = 1*positional_embedding_pre[i+keep_len]/4 + 3*positional_embedding_pre[i+1+keep_len]/4

    posisitonal_embedding_new[4*length -3*keep_len - 4] = positional_embedding_pre[length-1] + 0*(positional_embedding_pre[length-1] - positional_embedding_pre[length-2])/4
    posisitonal_embedding_new[4*length -3*keep_len - 3] = positional_embedding_pre[length-1] + 1*(positional_embedding_pre[length-1] - positional_embedding_pre[length-2])/4
    posisitonal_embedding_new[4*length -3*keep_len - 2] = positional_embedding_pre[length-1] + 2*(positional_embedding_pre[length-1] - positional_embedding_pre[length-2])/4
    posisitonal_embedding_new[4*length -3*keep_len - 1] = positional_embedding_pre[length-1] + 3*(positional_embedding_pre[length-1] - positional_embedding_pre[length-2])/4
            
    model.positional_embedding = nn.Parameter(posisitonal_embedding_new)

    return model

if __name__ == "__main__":
    encoder = get_longclip_embeds(_, device=torch.device("cuda:2"))
    encoder("Hello_world", "Hey baby")