# Copyright 2024  Kapil Devkota, Rohit Singh
# All rights reserved
# This code is available under the terms of the license available at https://github.com/rohitsinghlab/raygun
import argparse
import sys
sys.path.append("src/")
from b_raygun.model.raygun import Raygun 
from b_raygun.model.esmdecoder import DecoderBlock
from b_raygun.loader import RaygunData
from b_raygun.train_utils import train
from b_raygun.model.ltraygun import RaygunLightning
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import esm
import os
import pandas as pd
import itertools
import time
import json
from Bio.Align import substitution_matrices
import subprocess
import logging 
import lightning as L
import hydra
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import wandb
from datetime import datetime
from pathlib import Path

torch.set_float32_matmul_precision('high')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

@hydra.main(config_path="configs/", config_name="train",
            version_base = None)
def main(config: DictConfig):
    logger.info("Running Raygun training/finetuning...")
    config = OmegaConf.to_container(config, resolve=True)

    # create model and embedding folders
    os.makedirs(config["model_saveloc"], exist_ok = True)
    if config["esm2_embedding_saveloc"] is not None:
        os.makedirs(config["model_saveloc"], exist_ok = True)

    # Use ESM-2 650M
    esmmodel, esmalphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esmmodel              = esmmodel.to(0)
    esmmodel.eval()

    if config["log_wandb"]:
        wandb_logger = WandbLogger(project = "BATCH-TRAINING-RAYGUN")
    else:
        wandb_logger = None

    logger.info(f"Using pre-trained checkpoint.")
    _, _, hyparams = torch.hub.load('rohitsinghlab/raygun', 
                                    'pretrained_uniref50_95000_750M')
    
    model          = Raygun(numencoders = config["numencoders"],
                            numdecoders = config["numdecoders"],
                            esmdecodertotokenfile = "data/models/esm-decoder.sav",
                            esm_alphabet = esmalphabet.to_dict())    
    rayltmodule    = RaygunLightning(model, 
                                     esmalphabet,
                                     lr          = config["lr"],
                                     log_wandb   = config["log_wandb"],
#                                      save_dir    = config["model_saveloc"],
                                     traininglog = config["model_saveloc"] + "/error-log.txt")
    if "checkpoint" in config and config["checkpoint"] is not None:
        ckptpath   = Path(config["checkpoint"])
        checkpoint = torch.load(ckptpath, weights_only = True)
        if ckptpath.suffix == ".sav_original":
            rayltmodule.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            rayltmodule.load_state_dict(checkpoint["state_dict"])

    ## train and validation loaders
    traindata = RaygunData(fastafile = config["trainfasta"],
                           alphabet  = esmalphabet,
                           model     = esmmodel, 
                           device    = 0)
    trainloader = DataLoader(traindata, 
                             shuffle = False, 
                             batch_size = config["batch_size"],
                             collate_fn = traindata.collatefn)
    validdata = RaygunData(fastafile = config["validfasta"],
                           alphabet  = esmalphabet,
                           model     = esmmodel,
                           device    = 0)
    validloader = DataLoader(validdata, 
                            shuffle = False,
                            batch_size = config["batch_size"], 
                            collate_fn = validdata.collatefn)
    # Start the training
    
    ## checkpoint
    chk_callback = ModelCheckpoint(
                        monitor = "val_blosum_ratio",
                        mode    = "max",
                        save_top_k = config["num_to_save"], 
                        save_weights_only = True, 
                        dirpath = config["model_saveloc"],
                        filename = "model-{epoch:02d}-{val_blosum_ratio:.4f}"
                    )
    
    trainer = L.Trainer(logger = wandb_logger, 
                        callbacks = [chk_callback],
                        accelerator="gpu", 
                        devices=config["devices"], strategy="ddp",
                        max_epochs=config["epoch"], 
                        gradient_clip_val = config["clip"],
                        gradient_clip_algorithm = "value")
    
    trainer.fit(rayltmodule, trainloader, 
                validloader)
    return 

if __name__ == "__main__":
    main()
    
