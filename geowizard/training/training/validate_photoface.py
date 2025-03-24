# A reimplemented version in public environments by Xiao Fu and Mu Hu

import argparse
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn.parameter import Parameter

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import logging
import tqdm
import copy

import sys
sys.path.append("..")

from accelerate import Accelerator
import transformers
import numpy as np
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
import shutil

import diffusers
from diffusers import DiffusionPipeline, DDPMScheduler, DDIMScheduler, AutoencoderKL
from models.unet_2d_condition import UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
import accelerate

import cv2
from utils.de_normalized import align_scale_shift
from utils.depth2normal import *
from utils.train_validation import log_validation, log_photoface_validation
from utils.dataset_configuration import prepare_dataset, depth_scale_shift_normalization,  resize_max_res_tensor
from pathlib import Path
from PIL import Image
from torch.utils.tensorboard import SummaryWriter



# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.26.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def parse_args():
    parser = argparse.ArgumentParser(description="GeoWizard")

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    

    parser.add_argument(
        "--fined_tune_from_checkpoint",
        type=str,
        default=None,
        help="Path to the checkpoint to fine-tune from.",
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/data/",
        required=True,
        help="The Root Dataset Path.",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="photoface",
        help="The dataset name.",
    )

    parser.add_argument(
        "--csv_valid_path",
        type=str,
        default="/data/synthesis.csv",
        required=True,
        help="Path to train dataset csv"
    )
    
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

    parser.add_argument(
        "--recom_resolution",
        type=int,
        default=768,
        help=(
            "The resolution for resizeing the input images and the depth/disparity to make full use of the pre-trained model from \
                from the stable diffusion vae, for common cases, do not change this parameter"
        ),
    )
    # dataloaderes
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    
    parser.add_argument(
        "--output_valid_dir",
        type=str,
        default="valid",
        help="Validation directory.",
    )
    
    # get the local rank
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))

    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    args.dataset_path = ""
    return args

    
def main():

    ''' ------------------------Configs Preparation----------------------------'''
    # give the args parsers
    args = parse_args()
    
        # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    if args.seed is not None:
        set_seed(args.seed)

    ''' ------------------------Non-NN Modules Definition----------------------------'''
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder='scheduler')
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder='tokenizer')
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder='vae')
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder='text_encoder')
    unet = UNet2DConditionModel.from_pretrained(args.fined_tune_from_checkpoint, subfolder='unet_ema')
    noise_scheduler.timestep_spacing = "trailing"            
    val_mean, val_std, acc_list = log_photoface_validation(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        args=args,
        scheduler=noise_scheduler,
        epoch="validation",
    )
        


if __name__=="__main__":
    main()