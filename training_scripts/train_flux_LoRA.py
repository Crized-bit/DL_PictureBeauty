import argparse
import copy
import logging

logging.basicConfig(level = logging.INFO)


import math
import numpy as np
import PIL.Image as Image
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["FLUX_DEV_DEPTH_LORA"] = "/home/Diffusion_VK/weights/flux1-depth-dev-lora.safetensors"
# os.environ["FLUX_DEV"] = "~/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/0ef5fff789c832c5c7f4e127f94c8b54bbcced44/flux1-dev.safetensors"
# os.environ["FLUX_DEV"] = ""
from tqdm import tqdm
import shutil
from pathlib import Path
from torch import Tensor

import wandb
wandb.login()
import accelerate
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from safetensors.torch import save_file
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from datasets import load_from_disk
from packaging import version
from torchvision import transforms
from torchvision.transforms.functional import crop

from einops import rearrange, repeat
from torchao.quantization import quantize_, int8_weight_only, int4_weight_only
from flux.util import load_clip, load_ae, load_flow_model, load_t5, load_sft
from flux.sampling import prepare, denoise, get_noise, get_schedule
from flux.modules.lora import LinearLora

# from huggingface_hub import login
# login()

import diffusers
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, free_memory, compute_loss_weighting_for_sd3
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_torch_npu_available

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.32.0.dev0")

def prepare_control(
    t5,
    clip,
    img: Tensor,
    prompt: str | list[str],
    ae,
    img_cond_path: str,
) -> dict[str, Tensor]:
    # load and encode the conditioning image
    bs, _, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img_cond = Image.open(img_cond_path).convert("RGB")
    width = w * 8
    height = h * 8
    img_cond = img_cond.resize((width, height), Image.LANCZOS)
    img_cond = np.array(img_cond)
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

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )

logger = get_logger(__name__)
if is_torch_npu_available():
    torch.npu.config.allow_internal_format = False #type: ignore


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=1337, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=2, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=40)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=3,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=3,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", type=bool, default=True, help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--use_adafactor",
        action="store_true",
        help=(
            "Adafactor is a stochastic optimization method based on Adam that reduces memory usage while retaining"
            "the empirical benefits of adaptivity. This is achieved through maintaining a factored representation "
            "of the squared gradient accumulator across training steps."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="/home/Final_dataset_Games/",
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--image_column", 
        type=str, 
        default="result_picture", 
        help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="control_image",
        help="The column of the dataset containing the controlnet conditioning image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="caption",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default='A lone adventurer stands poised at the edge of a rugged cliff, her silhouette framed against a backdrop of towering rock formations shrouded in mist. The vast expanse of the foggy canyon below is broken by the sharp edges of weathered plateaus, their surfaces carved by time and the elements. Wisps of clouds swirl around the landscape, lending an ethereal, otherworldly quality to the scene, while sparse vegetation clings to the rocky terrain in the foreground, hinting at the harshness of the environment. She gazes ahead, embodying courage and determination, as though ready to uncover the secrets hidden beyond the horizon',
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default='/home/dataset/val/1.jpg',
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="flux_train_controlnet",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="logit_normal",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args


def get_train_dataset(args, accelerator):
    dataset = None
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_from_disk(
                args.dataset_name
            )
    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset.column_names #type: ignore

    # 6. Get the column names for input/target.
    if args.image_column is None:
        image_column = column_names[0]
        logger.info(f"image column defaulting to {image_column}")
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.caption_column is None:
        caption_column = column_names[1]
        logger.info(f"caption column defaulting to {caption_column}")
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"`--caption_column` value '{args.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.conditioning_image_column is None:
        conditioning_image_column = column_names[2]
        logger.info(f"conditioning image column defaulting to {conditioning_image_column}")
    else:
        conditioning_image_column = args.conditioning_image_column
        if conditioning_image_column not in column_names:
            raise ValueError(
                f"`--conditioning_image_column` value '{args.conditioning_image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    with accelerator.main_process_first():
        train_dataset = dataset.shuffle(seed=args.seed) #type: ignore
        if args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(args.max_train_samples))
    return train_dataset


def prepare_train_dataset(dataset, accelerator):
    image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[args.image_column]]
        conditioning_images = [image.convert("RGB") for image in examples[args.conditioning_image_column]]
        # image aug
        all_images = []
        all_conditioning_images = []
        train_crop = transforms.RandomCrop(args.resolution)
        for image, conditioning_image in zip(images, conditioning_images):
            image = image_transforms(image)
            conditioning_image = image_transforms(conditioning_image)

        
            y1, x1, h, w = train_crop.get_params(image, (args.resolution, args.resolution))
            image = crop(image, y1, x1, h, w)
            conditioning_image = crop(conditioning_image, y1, x1, h, w)

            all_images.append(image)
            all_conditioning_images.append(conditioning_image)

        examples["pixel_values"] = all_images
        examples["conditioning_pixel_values"] = all_conditioning_images
        examples["prompts"] = [prompt for prompt in examples[args.caption_column]]

        return examples

    with accelerator.main_process_first():
        dataset = dataset.with_transform(preprocess_train)

    return dataset


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    prompts = [example["prompts"] for example in examples]


    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "prompts": prompts,
    }


def main(args):
    logging_out_dir = Path(args.output_dir, args.logging_dir)

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=str(logging_out_dir))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS. A technique for accelerating machine learning computations on iOS and macOS devices.
    if torch.backends.mps.is_available():
        print("MPS is enabled. Disabling AMP.")
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        # DEBUG, INFO, WARNING, ERROR, CRITICAL
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizers
    # load clip tokenizer
    t5 = load_t5("cpu", max_length=512)
    clip = load_clip("cpu")
    ae = load_ae("flux-dev-depth-lora", device="cpu")

    model = load_flow_model("flux-dev-depth-lora", device="cpu")
    LoRA = load_sft(os.environ["FLUX_DEV_DEPTH_LORA"], device="cpu").keys()

    def filter_fn(module: torch.nn.Module, fqn: str) -> bool:
        if fqn in LoRA or not hasattr(module, "weight"):
            return False
        if module.weight is None:
            return False
        if isinstance(module, LinearLora):
            return True
        return True
    
    quantize_(model, int8_weight_only(), filter_fn=filter_fn)
    model.to(accelerator.device)

    

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        subfolder="scheduler",
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    ae.requires_grad_(False)
    model.requires_grad_(False)
    clip.requires_grad_(False)
    t5.requires_grad_(False)



    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                i = len(weights) - 1

                while len(weights) > 0:
                    weights.pop()
                    model = models[i]
                    
                    # todo: save the model
                    # sub_dir = "flux_controlnet"

                    lora_dict = {k:v for k,v in model.state_dict().items() if k in LoRA}
                    
                    save_file(lora_dict, os.path.join(output_dir, "lora_tile.safetensors"))
                    i -= 1

        def load_model_hook(models, input_dir):
            #todo: loading
            pass

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)


    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    lora_layers = []
    for name, param in model.named_parameters():
        if name in LoRA:
            param.requires_grad_()
            lora_layers.append(param)
            # print(name)
    params_to_optimize = lora_layers

    print_trainable_parameters(model=model)
    # use adafactor optimizer to save gpu memory
    if args.use_adafactor:
        from transformers import Adafactor

        optimizer = Adafactor(
            params_to_optimize,
            lr=args.learning_rate,
            scale_parameter=False,
            relative_step=False,
            # warmup_init=True,
            weight_decay=args.adam_weight_decay,
        )
    else:
        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    free_memory()

    dataset = get_train_dataset(args,
                                accelerator)
    
    # Then get the training dataset ready to be passed to the dataloader.
    train_dataset = prepare_train_dataset(dataset, accelerator) #type: ignore

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=num_training_steps_for_scheduler,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt")
        tracker_config.pop("validation_image")

        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    if True:
        t5.to(accelerator.device)
        clip.to(accelerator.device)
        ae.to(accelerator.device)

    wandb_tracker = accelerator.get_tracker("wandb")
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                model.train()
                # Convert images to latent space
                # vae encode
                pixel_values = batch["pixel_values"]
                img_cond = batch["conditioning_pixel_values"]
                bs, _, _, _ = pixel_values.shape

                with torch.no_grad():
                    pixel_latents = ae.encode(pixel_values).to(dtype=weight_dtype)
                    latent_cond = ae.encode(img_cond).to(dtype=weight_dtype)

                latent_cond = rearrange(latent_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)


                noise = torch.randn_like(pixel_latents).to(accelerator.device).to(dtype=weight_dtype)
                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bs,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=pixel_latents.device)

                # Add noise according to flow matching.
                sigmas = get_sigmas(timesteps, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype)
                noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise

                

                return_dict = prepare(t5, clip, noisy_model_input, batch["prompts"])

                return_dict["img_cond"] = latent_cond
                

                guidance_vec = torch.full((pixel_latents.shape[0],),
                                          10,
                                          device=accelerator.device,
                                          dtype=pixel_latents.dtype)
                    
                # with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                pred = model(
                    img=torch.cat((return_dict["img"], return_dict["img_cond"]), dim=-1).to(dtype=weight_dtype),
                    img_ids=return_dict["img_ids"].to(dtype=weight_dtype),
                    txt=return_dict["txt"].to(dtype=weight_dtype),
                    txt_ids=return_dict["txt_ids"].to(dtype=weight_dtype),
                    y=return_dict["vec"].to(dtype=weight_dtype),
                    timesteps=timesteps.to(dtype=weight_dtype),
                    guidance=guidance_vec.to(dtype=weight_dtype)
                    )
                
                # weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                loss = F.mse_loss(unpack(pred.float(), 1024, 1024), (noise - pixel_latents).float(), reduction="mean")
                accelerator.backward(loss)
                # Check if the gradient of each model parameter contains NaN
                for name, param in model.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        logger.error(f"Gradient for {name} contains NaN!")

                if accelerator.sync_gradients:
                    params_to_clip = model.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues.
                if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                        # prepare input
                        model.eval()
                        w, h = Image.open(args.validation_image).size
                        resize_by = 1024 / min(w, h)
                        w, h = int(w * resize_by) // 64 * 64, int(h * resize_by) // 64 * 64
                        
                        with torch.no_grad():
                            x = get_noise(
                                1,
                                h,
                                w,
                                device=accelerator.device,
                                dtype=torch.bfloat16,
                                seed=1337,
                            )

                            inp = prepare_control(
                                t5,
                                clip,
                                x,
                                prompt=args.validation_prompt,
                                ae=ae,
                                img_cond_path=args.validation_image)
                            
                            timesteps = get_schedule(50, inp["img"].shape[1], shift=True)

                        
                            x = denoise(model, **inp, timesteps=timesteps, guidance=10.)

                            x = unpack(x.float(), h, w)
                            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                                x = ae.decode(x)

                        x = x.clamp(-1, 1)
                        x = rearrange(x[0], "c h w -> h w c")
                        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
                            # tracker.writer.add_images(args.validation_prompt, img, step, dataformats="NHWC")
                        wandb_tracker.log({"val_photo_lara": wandb.Image(img)}, step=global_step)

                        img.save('./test_img.jpg')
                        torch.cuda.empty_cache()

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        lora_dict = {k:v for k,v in model.state_dict().items() if k in LoRA}
        save_file(lora_dict, os.path.join("/home", "lora_tile.safetensors"))
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)